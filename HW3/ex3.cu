/* CUDA 10.2 has a bug that prevents including <cuda/atomic> from two separate
 * object files. As a workaround, we include ex2.cu directly here. */
#include "ex2.cu"

#include <cassert>
#include <vector>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <infiniband/verbs.h>


class server_rpc_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;
    std::array<int, OUTSTANDING_REQUESTS> input_read_done_count;


public:
    explicit server_rpc_context(uint16_t tcp_port) : rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
        std::fill(input_read_done_count.begin(), input_read_done_count.end(), 0);

    }

    virtual void event_loop() override
    {
        /* so the protocol goes like this:
         * 1. we'll wait for a CQE indicating that we got an Send request from the client.
         *    this tells us we have new work to do. The wr_id we used in post_recv tells us
         *    where the request is.
         * 2. now we send an RDMA Read to the client to retrieve the request.
         *    we will get a completion indicating the read has completed.
         * 3. we process the request on the GPU.
         * 4. upon completion, we send an RDMA Write with immediate to the client with
         *    the results.
         */
        rpc_request* req;
        uchar *img_target, *img_reference;
        uchar *img_out;

        bool terminate = false, got_last_cqe = false;

        while (!terminate || !got_last_cqe) {
            // Step 1: Poll for CQE
            struct ibv_wc wc;
            int ncqes = ibv_poll_cq(cq, 1, &wc);
            if (ncqes < 0) {
                perror("ibv_poll_cq() failed");
                exit(1);
            }
            if (ncqes > 0) {
		VERBS_WC_CHECK(wc);

                switch (wc.opcode) {
                case IBV_WC_RECV:
                    /* Received a new request from the client */
                    req = &requests[wc.wr_id];
                    img_target = &images_target[wc.wr_id * IMG_BYTES];
                    img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                    /* Terminate signal */
                    if (req->request_id == -1) {
                        printf("Terminating...\n");
                        terminate = true;
                        goto send_rdma_write;
                    }

                    /* Step 2: send RDMA Read to client to read the input */
                    input_read_done_count[wc.wr_id] = 0;
                    post_rdma_read(
                        img_target,                 // local_src
                        req->input_target_length,   // len
                        mr_images_target->lkey,     // lkey
                        req->input_target_addr,     // remote_dst
                        req->input_target_rkey,     // rkey
                        wc.wr_id);                  // wr_id

                    post_rdma_read(
                        img_reference,              // local_src
                        req->input_reference_length,// len
                        mr_images_reference->lkey,  // lkey
                        req->input_reference_addr,  // remote_dst
                        req->input_reference_rkey,  // rkey
                        wc.wr_id);                  // wr_id
                break;

                case IBV_WC_RDMA_READ:
                    /* Completed RDMA read for a request */
                    input_read_done_count[wc.wr_id]++;
                    if (input_read_done_count[wc.wr_id] == 2){
                        req = &requests[wc.wr_id];
                        img_target = &images_target[wc.wr_id * IMG_BYTES];
                        img_reference = &images_reference[wc.wr_id * IMG_BYTES];
                        img_out = &images_out[wc.wr_id * IMG_BYTES];

                        // Step 3: Process on GPU
                        while(!gpu_context->enqueue(wc.wr_id, img_target, img_reference, img_out)){};
                    }
		        break;
                    
                case IBV_WC_RDMA_WRITE:
                    /* Completed RDMA Write - reuse buffers for receiving the next requests */
                    post_recv(wc.wr_id % OUTSTANDING_REQUESTS);

                    if (terminate)
                    got_last_cqe = true;

                break;
                default:
                    printf("Unexpected completion\n");
                    assert(false);
                }
            }

            // Dequeue completed GPU tasks
            int dequeued_job_id;
            if (gpu_context->dequeue(&dequeued_job_id)) {
                req = &requests[dequeued_job_id];
                img_out = &images_out[dequeued_job_id * IMG_BYTES];

send_rdma_write:
                // Step 4: Send RDMA Write with immediate to client with the response
		post_rdma_write(
                    req->output_addr,                       // remote_dst
                    terminate ? 0 : req->output_length,     // len
                    req->output_rkey,                       // rkey
                    terminate ? 0 : img_out,                // local_src
                    mr_images_out->lkey,                    // lkey
                    dequeued_job_id + OUTSTANDING_REQUESTS, // wr_id
                    (uint32_t*)&req->request_id);           // immediate
            }
        }
    }
};

class client_rpc_context : public rdma_client_context {
private:
    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

    struct ibv_mr *mr_images_target, *mr_images_reference; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
public:
    explicit client_rpc_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
    }

    ~client_rpc_context()
    {
        kill();
    }

    virtual void set_input_images(uchar *images_target, uchar* images_reference, size_t bytes) override
    {
        /* register a memory region for the input images. */
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, IBV_ACCESS_REMOTE_READ);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        /* register a memory region for the output images. */
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for output images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        if ((requests_sent - send_cqes_received) == OUTSTANDING_REQUESTS)
            return false;

        struct ibv_sge sg; /* scatter/gather element */
        struct ibv_send_wr wr; /* WQE */
        struct ibv_send_wr *bad_wr; /* ibv_post_send() reports bad WQEs here */

        /* step 1: send request to server using Send operation */
        
        struct rpc_request *req = &requests[requests_sent % OUTSTANDING_REQUESTS];
        req->request_id = job_id;
        req->input_target_rkey = target ? mr_images_target->rkey : 0;
        req->input_target_addr = (uintptr_t)target;
        req->input_target_length = IMG_BYTES;
        req->input_reference_rkey = reference ? mr_images_reference->rkey : 0;
        req->input_reference_addr = (uintptr_t)reference;
        req->input_reference_length = IMG_BYTES;
        req->output_rkey = result ? mr_images_out->rkey : 0;
        req->output_addr = (uintptr_t)result;
        req->output_length = IMG_BYTES;

        /* RDMA send needs a gather element (local buffer)*/
        memset(&sg, 0, sizeof(struct ibv_sge));
        sg.addr = (uintptr_t)req;
        sg.length = sizeof(*req);
        sg.lkey = mr_requests->lkey;

        /* WQE */
        memset(&wr, 0, sizeof(struct ibv_send_wr));
        wr.wr_id = job_id; /* helps identify the WQE */
        wr.sg_list = &sg;
        wr.num_sge = 1;
        wr.opcode = IBV_WR_SEND;
        wr.send_flags = IBV_SEND_SIGNALED; /* always set this in this excersize. generates CQE */

        /* post the WQE to the HCA to execute it */
        if (ibv_post_send(qp, &wr, &bad_wr)) {
            perror("ibv_post_send() failed");
            exit(1);
        }

        ++requests_sent;

        return true;
    }

    virtual bool dequeue(int *job_id) override
    {
        /* When WQE is completed we expect a CQE */
        /* We also expect a completion of the RDMA Write with immediate operation from the server to us */
        /* The order between the two is not guarenteed */

        struct ibv_wc wc; /* CQE */
        int ncqes = ibv_poll_cq(cq, 1, &wc);
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        if (ncqes == 0)
            return false;

	VERBS_WC_CHECK(wc);

        switch (wc.opcode) {
        case IBV_WC_SEND:
            ++send_cqes_received;
            return false;
        case IBV_WC_RECV_RDMA_WITH_IMM:
            *job_id = wc.imm_data;
            break;
        default:
            printf("Unexpected completion type\n");
            assert(0);
        }

        /* step 2: post receive buffer for the next RPC call (next RDMA write with imm) */
        post_recv();

        return true;
    }

    void kill()
    {
        while (!enqueue(-1, // Indicate termination
                       NULL, NULL, NULL)) ;
        int job_id = 0;
        bool dequeued;
        do {
            dequeued = dequeue(&job_id);
        } while (!dequeued || job_id != -1);
    }
};

struct queues_parameters {
    uint32_t cpu_to_gpu_mr_rkey;
    queue<cpu_to_gpu_queues_entry>* cpu_to_gpu_mr_addr;
    uint32_t gpu_to_cpu_mr_rkey;
    queue<gpu_to_cpu_queues_entry>* gpu_to_cpu_mr_addr;
    uint32_t mr_images_target_rkey;
    uchar* mr_images_target_addr;
    uint32_t mr_images_reference_rkey;
    uchar* mr_images_reference_addr;
    uint32_t mr_images_out_rkey;
    uchar* mr_images_out_addr;

    int number_of_queues;
};

class server_queues_context : public rdma_server_context {
private:
    std::unique_ptr<queue_server> gpu_context;

    /* TODO: add memory region(s) for CPU-GPU queues */
    struct ibv_mr* cpu_to_gpu_mr;
    struct ibv_mr* gpu_to_cpu_mr;

    // Maybe I need to add the Requests as MR
    

public:
    explicit server_queues_context(uint16_t tcp_port) :
        rdma_server_context(tcp_port),
        gpu_context(create_queues_server(256))
    {
        /* TODO Initialize additional server MRs as needed. */
        queue<cpu_to_gpu_queues_entry> *cpu_to_gpu_queues = gpu_context->cpu_to_gpu_queues;
        queue<gpu_to_cpu_queues_entry> *gpu_to_cpu_queues = gpu_context->gpu_to_cpu_queues;
        int blocks = gpu_context->blocks;
        
        cpu_to_gpu_mr = ibv_reg_mr(pd, cpu_to_gpu_queues, blocks * sizeof(queue<cpu_to_gpu_queues_entry>), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!cpu_to_gpu_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        gpu_to_cpu_mr = ibv_reg_mr(pd, gpu_to_cpu_queues, blocks * sizeof(queue<gpu_to_cpu_queues_entry>), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!gpu_to_cpu_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        /* TODO Exchange rkeys, addresses, and necessary information (e.g.
         * number of queues) with the client */
        struct queues_parameters gpu_queues_parameters = 
                {cpu_to_gpu_mr->rkey,
                cpu_to_gpu_queues,
                gpu_to_cpu_mr->rkey,
                gpu_to_cpu_queues,
                mr_images_target->rkey,
                images_target,
                mr_images_reference->rkey,
                images_reference,
                mr_images_out->rkey,
                images_out,
                blocks};

        send_over_socket(&gpu_queues_parameters, sizeof(struct queues_parameters));
    }

    ~server_queues_context()
    {
        /* TODO destroy the additional server MRs here */
        ibv_dereg_mr(cpu_to_gpu_mr);
        ibv_dereg_mr(gpu_to_cpu_mr);
    }

    virtual void event_loop() override
    {
        /* TODO simplified version of server_rpc_context::event_loop. As the
         * client use one sided operations, we only need one kind of message to
         * terminate the server at the end. */
        bool terminate_server = false;
        while(terminate_server == false) {
            recv_over_socket(&terminate_server, sizeof(bool));
        }
    }
};

uint64_t atomic_sizet_size = sizeof(cuda::atomic<size_t>);

class client_queues_context : public rdma_client_context {
private:
    /* TODO add necessary context to track the client side of the GPU's
     * producer/consumer queues */
    struct queues_parameters gpu_queues_parameters;

    struct ibv_mr *mr_images_target, *mr_images_reference; /* Memory region for input images */
    struct ibv_mr *mr_images_out; /* Memory region for output images */
    /* TODO define other memory regions used by the client here */
    queue<cpu_to_gpu_queues_entry> cpu_to_gpu_ring_buff;
    struct ibv_mr* cpu_to_gpu_ring_buff_mr;
    queue<gpu_to_cpu_queues_entry> gpu_to_cpu_ring_buff;
    struct ibv_mr* gpu_to_cpu_ring_buff_mr;

    cpu_to_gpu_queues_entry ctg_request;
    struct ibv_mr* ctg_request_mr;
    gpu_to_cpu_queues_entry gtc_request;
    struct ibv_mr* gtc_request_mr;

    int next_block_enqueue = 1;
    int next_block_dequeue = 0;

    uint32_t requests_sent = 0;
    uint32_t send_cqes_received = 0;

public:
    client_queues_context(uint16_t tcp_port) : rdma_client_context(tcp_port)
    {
        /* TODO communicate with server to discover number of queues, necessary
         * rkeys / address, or other additional information needed to operate
         * the GPU queues remotely. */
        recv_over_socket(&gpu_queues_parameters, sizeof(struct queues_parameters));

        /* TODO register memory regions for CPU-GPU queues */
        cpu_to_gpu_ring_buff_mr = ibv_reg_mr(pd, &cpu_to_gpu_ring_buff, sizeof(queue<cpu_to_gpu_queues_entry>), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!cpu_to_gpu_ring_buff_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        gpu_to_cpu_ring_buff_mr = ibv_reg_mr(pd, &gpu_to_cpu_ring_buff, sizeof(queue<gpu_to_cpu_queues_entry>), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!gpu_to_cpu_ring_buff_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        ctg_request_mr = ibv_reg_mr(pd, &ctg_request, sizeof(cpu_to_gpu_queues_entry), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!ctg_request_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }

        gtc_request_mr = ibv_reg_mr(pd, &gtc_request, sizeof(gpu_to_cpu_queues_entry), IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!gtc_request_mr) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    ~client_queues_context()
    {
        /* TODO terminate the server and release memory regions and other resources */
        bool terminate_server = true;
        send_over_socket(&terminate_server, sizeof(bool));
        ibv_dereg_mr(cpu_to_gpu_ring_buff_mr);
        ibv_dereg_mr(gpu_to_cpu_ring_buff_mr);
        ibv_dereg_mr(ctg_request_mr);
        ibv_dereg_mr(gtc_request_mr);
        ibv_dereg_mr(mr_images_target);
        ibv_dereg_mr(mr_images_reference);
        ibv_dereg_mr(mr_images_out);
    }

    virtual void set_input_images(uchar *images_target, uchar* images_reference, size_t bytes) override
    {
        // TODO register memory
        mr_images_target = ibv_reg_mr(pd, images_target, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_target) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
        mr_images_reference = ibv_reg_mr(pd, images_reference, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_reference) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual void set_output_images(uchar *images_out, size_t bytes) override
    {
        // TODO register memory
        mr_images_out = ibv_reg_mr(pd, images_out, bytes, IBV_ACCESS_REMOTE_READ | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
        if (!mr_images_out) {
            perror("ibv_reg_mr() failed for input images");
            exit(1);
        }
    }

    virtual bool enqueue(int job_id, uchar *target, uchar *reference, uchar *result) override
    {
        if (requests_sent - send_cqes_received == OUTSTANDING_REQUESTS)
        {
            return false;
        }
        /* TODO use RDMA Write and RDMA Read operations to enqueue the task on
         * a CPU-GPU producer consumer queue running on the server. */
        struct ibv_wc wc; /* CQE */
        int ncqes;

        wc.wr_id = 11;

        uint64_t ring_buffer = ((uint64_t)gpu_queues_parameters.cpu_to_gpu_mr_addr + (next_block_enqueue * sizeof(queue<cpu_to_gpu_queues_entry>)));
        next_block_enqueue = (next_block_enqueue + 1) % gpu_queues_parameters.number_of_queues;
        
        post_rdma_read(
            &cpu_to_gpu_ring_buff,                      // local_src
            sizeof(queue<cpu_to_gpu_queues_entry>),     // len
            cpu_to_gpu_ring_buff_mr->lkey,              // lkey
            ring_buffer,                                // remote_dst
            gpu_queues_parameters.cpu_to_gpu_mr_rkey,   // rkey
            wc.wr_id                                    // wr_id
        );
        
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);

        if(cpu_to_gpu_ring_buff.is_full())
        {
            return false;
        }

        wc.wr_id = 12;

        uint64_t target_image_r_addr = (uint64_t)gpu_queues_parameters.mr_images_target_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);
        post_rdma_write(
            target_image_r_addr,                            // remote_dst
            IMG_BYTES,                                      // len
            gpu_queues_parameters.mr_images_target_rkey,    // rkey
            target,                                         // local_src
            mr_images_target->lkey,                         // lkey
            wc.wr_id                                        // wr_id
                                                            // immediate
        );                                                  

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);

        wc.wr_id = 13;

        uint64_t reference_image_r_addr = (uint64_t)gpu_queues_parameters.mr_images_reference_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);
        post_rdma_write(
            reference_image_r_addr,                         // remote_dst
            IMG_BYTES,                                      // len
            gpu_queues_parameters.mr_images_reference_rkey, // rkey
            reference,                                      // local_src
            mr_images_reference->lkey,                      // lkey
            wc.wr_id                                        // wr_id
                                                            // immediate
        );

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);
        
        uint64_t output_image_r_addr = (uint64_t)gpu_queues_parameters.mr_images_out_addr + ((job_id % OUTSTANDING_REQUESTS) * IMG_BYTES);
        
        ctg_request.job_id = job_id;
        ctg_request.target = (uchar*)(target_image_r_addr);
        ctg_request.reference = (uchar*)(reference_image_r_addr);
        ctg_request.img_out = (uchar*)(output_image_r_addr);
        ctg_request.remote_img_out = (uchar*)(result);

        cpu_to_gpu_ring_buff.push(&ctg_request);

        wc.wr_id = 15;

        post_rdma_write(
            ring_buffer,                                // remote_dst
            sizeof(queue<cpu_to_gpu_queues_entry>),     // len
            gpu_queues_parameters.cpu_to_gpu_mr_rkey,   // rkey
            &cpu_to_gpu_ring_buff,                      // local_src
            cpu_to_gpu_ring_buff_mr->lkey,              // lkey
            wc.wr_id                                    // wr_id
                                                        // immediate
        );

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);

        requests_sent++;

        return true;
    }

    virtual bool dequeue(int *img_id) override
    {
        /* TODO use RDMA Write and RDMA Read operations to detect the completion and dequeue a processed image
         * through a CPU-GPU producer consumer queue running on the server. */
        struct ibv_wc wc; /* CQE */
        int ncqes;

        wc.wr_id = 21;

        uint64_t ring_buffer = (uint64_t)gpu_queues_parameters.gpu_to_cpu_mr_addr + (next_block_dequeue * sizeof(queue<gpu_to_cpu_queues_entry>));
        next_block_dequeue = (next_block_dequeue + 1) % gpu_queues_parameters.number_of_queues;

        post_rdma_read(
            &gpu_to_cpu_ring_buff,                      // local_src
            sizeof(queue<gpu_to_cpu_queues_entry>),     // len
            gpu_to_cpu_ring_buff_mr->lkey,              // lkey
            ring_buffer,                                // remote_dst
            gpu_queues_parameters.gpu_to_cpu_mr_rkey,   // rkey
            wc.wr_id                                    // wr_id
        );
        
        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }
        VERBS_WC_CHECK(wc);

        if(gpu_to_cpu_ring_buff.is_empty())
        {
            return false;
        }
        
        gpu_to_cpu_ring_buff.pop(&gtc_request);

        wc.wr_id = 23;
        uint64_t output_image_r_addr = (uint64_t)gtc_request.img_out;
        post_rdma_read(
            (void*)gtc_request.remote_img_out,              // local_src
            IMG_BYTES,                                      // len
            mr_images_out->lkey,                            // lkey
            output_image_r_addr,                            // remote_dst
            gpu_queues_parameters.mr_images_out_rkey,       // rkey
            wc.wr_id                                        // wr_id
        );

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }

        wc.wr_id = 24;
        
        post_rdma_write(
            ring_buffer,                                // remote_dst
            sizeof(queue<gpu_to_cpu_queues_entry>),     // len
            gpu_queues_parameters.gpu_to_cpu_mr_rkey,   // rkey
            &gpu_to_cpu_ring_buff,                      // local_src
            gpu_to_cpu_ring_buff_mr->lkey,              // lkey
            wc.wr_id                                    // wr_id
                                                        // immediate
        );

        while (( ncqes = ibv_poll_cq(cq, 1, &wc)) == 0) { }
        if (ncqes < 0) {
            perror("ibv_poll_cq() failed");
            exit(1);
        }

        send_cqes_received++;
        *img_id = gtc_request.job_id;
#ifdef DEBUG
        printf("Dequeued job\n");
#endif
        return true;
    }
};

std::unique_ptr<rdma_server_context> create_server(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<server_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<server_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}

std::unique_ptr<rdma_client_context> create_client(mode_enum mode, uint16_t tcp_port)
{
    switch (mode) {
    case MODE_RPC_SERVER:
        return std::make_unique<client_rpc_context>(tcp_port);
    case MODE_QUEUE:
        return std::make_unique<client_queues_context>(tcp_port);
    default:
        printf("Unknown mode.\n");
        exit(1);
    }
}
