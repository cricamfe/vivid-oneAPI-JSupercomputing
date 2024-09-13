#pragma once
#ifndef CIRCULAR_BUFFER_HPP_
#define CIRCULAR_BUFFER_HPP_

#include <vector>
#include "pipeline_template.hpp"

using namespace Pipeline_template;
using namespace oneapi;

class circular_buffer {
public:
    explicit circular_buffer(size_t size_, FloatBuffer* global_f, FloatBuffer* global_c, int n_filters, sycl::queue &Q)
        : size{size_}, buf{std::vector<ViVidItem*>(size_)}, global_frame{global_f}, global_cla{global_c}, num_filters{n_filters}, bufferQueue{Q} 
    {
        for(size_t i=0; i<size_; ++i) {
            buf[i] = new ViVidItem(i, global_frame, global_cla, num_filters, bufferQueue);
        }
        write_pos = size;
    }

    ~circular_buffer() {
        for(auto& item : buf) {
            delete item;
        }
    }

    ViVidItem* get() {
        if(read_pos >= write_pos) {
            // Buffer is empty, cannot get
            std::cerr << "Buffer is empty, cannot get\n";
            return nullptr;
        }
        auto item = buf[read_pos % size];
        ++read_pos;
        return item;
    }

    void recycle(ViVidItem* item) {
        if(write_pos >= read_pos + size) {
            // Buffer is full, cannot recycle
            std::cerr << "Buffer is full, cannot recycle\n";
            return;
        }
        item->recycle();
        buf[write_pos % size] = item;
        ++write_pos;
    }

	void reset() noexcept {
		read_pos = write_pos = 0;
	}

	size_t free_space() noexcept {
		if (write_pos < read_pos) {
			return read_pos - write_pos - 1;
		} else {
			return (read_pos + size - write_pos - 1) % size;
		}
	}

	size_t capacity() noexcept {
		return size;
	}

	bool is_full() noexcept {
		return ((write_pos + 1) % size) == read_pos;
	}

private:
    size_t size;					//< Size of the buffer
    std::vector<ViVidItem*> buf;	//< Vector of ViVidItem objects
    size_t read_pos = 0;  			//< Position of the next item to read (assign to a thread)
    size_t write_pos = 0; 			//< Position of the next slot to write to (recycle an item)
    FloatBuffer* global_frame; 		//< Constant buffer read only
    FloatBuffer* global_cla; 		//< Constant buffer read only
    sycl::queue &bufferQueue;		//< SYCL queue object used for memory management
    int num_filters;				//< Number of filters in the pipeline
};

#endif