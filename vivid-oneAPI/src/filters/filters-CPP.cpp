# include "filters-CPP.hpp"

/******************
* Filters
******************/
float * transposeBank(float* filter_bank) {
    // reorganize data in SIMD8 vectors
    // |0 1 2 .. 8| 0 1 2 .. 8 ..  =>> 0 0 0 ... 1 1 1 ..
    int filter_size = 9;
    int num_filters = 100;
    float* tmpbank = (float*)malloc(num_filters * filter_size*sizeof(float)); // __malloc()
    for(int i=0; i<num_filters/8; i++)
    {
        for(int j=0; j<9; j++) {
            for(int k=0; k<8; k++)
                tmpbank[i*8*9+ j*8+ k] = filter_bank[i*8*9+ j+ k*9];
        }
    }
    // leftovers in smaller vecs

    {
        for(int j=0; j<9; j++) {
            for(int k=0; k<4; k++)
                tmpbank[96*9 + j*4+ k] = filter_bank[96*9 + j+ k*9];
        }
    }
    return tmpbank;
}
//-----------------------------------------------------------------
void cosine_filter_transpose(float* fr_data, float* ind, float *val, float* fb_array_main, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch)
{
    float * fb_array = transposeBank(fb_array_main);
    //do convolution
    const int apron_y = filter_h / 2;
    const int apron_x = filter_w / 2;

    const int filter_size = filter_h * filter_w;

    int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

    int oi = 0;
    for (int ii=-apron_y; ii<=apron_y; ii++) {
        for (int jj=-apron_y; jj<=apron_y; jj++) {
            pixel_offsets[oi] = ii * width + jj;
            oi++;
        }
    }
    // 100 filters, each 9 values
    int n_threads = 1;

    int valid_height = height - 2 * apron_y;
    int height_step = valid_height / n_threads + 1;

    for (int tid=0; tid<n_threads; tid++) {
        int start_y = apron_y + tid * height_step;
        int end_y = min(start_y + height_step, height - apron_y);
    
		//-------------------------------run CG
		float *image_cache = (float*) std::aligned_alloc(32, sizeof(float) * filter_size);

		for (int i=start_y; i<end_y; i++) {
            float* fr_ptr = fr_data + i * width + apron_x;
			float* ass_out = ind + i * pitch/sizeof(float) + apron_x;   // modified to get output in two separated arrays
			float* wgt_out = val + i * pitch/sizeof(float) + apron_x;


			for (int j=apron_x; j<(width - apron_x); j++ ) {
				for (int ii=0; ii< filter_size; ii++) {
					// copy each pixel to all elements of vector
					image_cache[ii] = fr_ptr[pixel_offsets[ii]];
				}

				float max_sim = -1e6;
				int best_ind = -1;
				int fi=0;
				int filter_ind = 0;
				int sssize = 9;
				// 96 filters, 9 values each
				while (fi<((n_filters/8)*8)*filter_size)
				{
					float temp_sum[8] = {0,0,0,0,0,0,0,0};
					for(int i=0; i<sssize; i++) {
						float img = image_cache[i];
						for(int j=0; j<8; j++) {
                            temp_sum[j] += img * fb_array[fi++];
						}
					}
					for(int j=0; j<8; j++) {
						temp_sum[j] = fabs(temp_sum[j]);
					}
					for(int j=0; j<8; j++) {
						if(temp_sum[j] > max_sim) {
							max_sim = temp_sum[j];
							best_ind = filter_ind+j;
						}
					}
					filter_ind += 8;
				}
				float temp_sum[4] = {0,0,0,0};
				for(int i=0; i<9; i++) {
					#pragma ivdep
					for(int j=0; j<4; j++) {
                        temp_sum[j] += image_cache[i] * fb_array[fi++];
					}
				}
				for(int j=0; j<4; j++) {
					temp_sum[j] = fabs(temp_sum[j]);
				}
				for(int j=0; j<4; j++) {
					if(temp_sum[j] > max_sim) {
						max_sim = temp_sum[j];
						best_ind = filter_ind+j;
					}
				}

				*ass_out = (float)best_ind;
				*wgt_out = max_sim;

				fr_ptr++;
				ass_out++;
				wgt_out++;
			}
		}
		std::free(image_cache);
    }
    free(fb_array);
    free(pixel_offsets); //added by andres, I think it is necessary
}

/**************************************
 * Filter 2 cpu
 * *************************/
void block_histogram(float *ptr_his, float *ptr_ind, float *ptr_val, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind) {
    //variables
    int n_parts_y = (im_height-2) / cell_size;
    int n_parts_x = (im_width-2) / cell_size;
    int start_i = 1;
    int start_j = 1;
    //end variables

    for (int write_i=0; write_i<n_parts_y; write_i++) {
        for (int write_j=0; write_j<n_parts_x; write_j++) {
            int out_ind = (write_i*n_parts_x + write_j) * pitch_his;
            int read_i = (start_i + (write_i * cell_size)) * pitch_ind;
            for (int i=0; i<cell_size; i++) {
                int read_j = start_j + write_j * cell_size ;

                for (int j=0; j<cell_size; j++) {
                    int bin_ind = (int)ptr_ind[read_i+read_j+j];
                    float weight = ptr_val[read_i+read_j+j];
                    ptr_his[out_ind + bin_ind] += weight;
                }
                read_i += pitch_ind;
            }
        }
    }
}


/*****************************************
 * Filter 3 CPU
 * ***************************/
void pwdist_c(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth){
    for (int i=0; i<aheight; i++) {
		for (int j=0; j<bheight; j++) {
			float sum = 0.0;

			for (size_t k = 0; k < adatawidth; k++) {
				float dif = (ptra[i*awidth+k] - ptrb[j*awidth+k]);
                sum += dif*dif;
			}
			out_data[i*owidth+j]=sum;
		}
	}
}
