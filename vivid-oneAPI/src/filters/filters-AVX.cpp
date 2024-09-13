#include "filters-AVX.hpp"
// *********************************************************************************************************************
// *  FILTER 1: AVX2 implementation of the cosine filter
// *********************************************************************************************************************
void cosine_filter_AVX(float* fr_data, float* ind, float *val, float* fb_array, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch) {
	//do convolution
	const int apron_y = filter_h / 2;
	const int apron_x = filter_w / 2;

	const int filter_size = filter_h * filter_w;

	const int filter_bank_size = filter_size * n_filters;

	int *pixel_offsets=(int*) malloc(sizeof(int)*filter_size);

	int oi = 0;
	for (int ii=-apron_y; ii<=apron_y; ii++){
		for (int jj=-apron_y; jj<=apron_y; jj++){
			pixel_offsets[oi] = ii * width + jj;
			oi++;
		}
	}
	// 100 filters, each 9 values
	int imask = 0x7fffffff;
	float fmask = *((float*)&imask);
	int n_threads =1;

	int valid_height = height - 2 * apron_y;
	int height_step = valid_height / n_threads + 1;
	
	for (int tid=0; tid<n_threads; tid++){
		int start_y = apron_y + tid * height_step;
		int end_y = std::min(start_y + height_step, height - apron_y);
			for (int i=start_y; i<end_y; i++){
				float* fr_ptr = fr_data + i * width + apron_x;
				float* ass_out = ind + i * pitch/sizeof(float) + apron_x;   // modified to get output in two separated arrays
				float* wgt_out = val + i * pitch/sizeof(float) + apron_x;

				for (int j=apron_x; j<(width - apron_x); j++ ){
					__m256 image_cache0 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[0]]);
					__m256 image_cache1 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[1]]);
					__m256 image_cache2 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[2]]);
					__m256 image_cache3 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[3]]);
					__m256 image_cache4 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[4]]);
					__m256 image_cache5 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[5]]);
					__m256 image_cache6 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[6]]);
					__m256 image_cache7 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[7]]);
					__m256 image_cache8 = _mm256_broadcast_ss(&fr_ptr[pixel_offsets[8]]);

					float max_sim[8] = {-1e6, 
						-1e6, -1e6, -1e6, -1e6, -1e6, -1e6, -1e6};
					int best_ind = -1;

					int fi=0;
					int filter_ind = 0;

					// 96 filters, 9 values each
					while (fi<((n_filters/8)*8)*filter_size) {
						__m256 temp_sum = _mm256_set1_ps(0.0f);

						// no fused multiply add :(
						// current value of 8 filters
						__m256 curr_filter = _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache0, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache1, curr_filter), temp_sum);

						curr_filter = _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache2, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache3, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache4, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache5, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache6, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache7, curr_filter), temp_sum);

						curr_filter= _mm256_load_ps(&fb_array[fi]);
						fi+=8;
						temp_sum = _mm256_add_ps(_mm256_mul_ps(image_cache8, curr_filter), temp_sum);

						// calculating absolute value by clearing the last digit
						__m256 mask = _mm256_set1_ps(fmask);

						temp_sum = _mm256_and_ps(mask, temp_sum);

						__m256 max_fil = _mm256_load_ps(max_sim);
						// code 14

						int r;

						// low 128 half
						// copy low to high
						__m256 temp_sum2 = _mm256_insertf128_ps(temp_sum,
							_mm256_extractf128_ps(temp_sum, 0), 1);
						__m256 cpm = _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS);
						r = _mm256_movemask_ps(cpm);


						if(r&(1<<0)) {
							best_ind = filter_ind+7;
							const int control = 0;
							max_fil = _mm256_permute_ps(temp_sum2, 0b0); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<1)) {
							best_ind = filter_ind+6;
							const int control = 1|(1<<2)|(1<<4)|(1<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<2)) {
							best_ind = filter_ind+5;
							const int control = 2|(2<<2)|(2<<4)|(2<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<3)) {
							best_ind = filter_ind+4;
							const int control = 3|(3<<2)|(3<<4)|(3<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						// high 128 half
						// copy high to low
						temp_sum2 = _mm256_insertf128_ps(temp_sum,
							_mm256_extractf128_ps(temp_sum, 1), 0);
						cpm = _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS);
						r = _mm256_movemask_ps(cpm);


						if(r&(1<<0)) {
							best_ind = filter_ind+3;
							const int control = 0;
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<1)) {
							best_ind = filter_ind+2;
							const int control = 1|(1<<2)|(1<<4)|(1<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<2)) {
							best_ind = filter_ind+1;
							const int control = 2|(2<<2)|(2<<4)|(2<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}

						if(r&(1<<3)) {
							best_ind = filter_ind+0;
							const int control = 3|(3<<2)|(3<<4)|(3<<6);
							max_fil = _mm256_permute_ps(temp_sum2, control); 
							r=_mm256_movemask_ps( _mm256_cmp_ps(temp_sum2, max_fil, _CMP_GT_OS));
						}


						_mm256_store_ps(max_sim, max_fil);
						// printf("max1 :%f\n", max_fil.m128_f32[0]);


						filter_ind += 8;
					}

					// leftover filters
					__m128 temp_sum = _mm_set1_ps(0.0f);


					// current value of 4 filters
					__m128 curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache0, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache1, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache2, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache3, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache4, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache5, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache6, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache7, 0), curr_filter), temp_sum);

					curr_filter = _mm_load_ps(&fb_array[fi]);
					fi+=4;
					temp_sum = _mm_add_ps(_mm_mul_ps(_mm256_extractf128_ps(image_cache8, 0), curr_filter), temp_sum);

					__m128 max_fil = _mm_load_ss(max_sim);

					__m128 cpm = _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS);
					int	r = _mm_movemask_ps(cpm);


					if(r&(1<<0)) {
						best_ind = filter_ind+3;
						const int control = 0;
						max_fil = _mm_permute_ps(temp_sum, control); 
						r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
					}

					if(r&(1<<1)) {
						best_ind = filter_ind+2;
						const int control = 1|(1<<2)|(1<<4)|(1<<6);
						max_fil = _mm_permute_ps(temp_sum, control); 
						r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
					}

					if(r&(1<<2)) {
						best_ind = filter_ind+1;
						const int control = 2|(2<<2)|(2<<4)|(2<<6);
						max_fil = _mm_permute_ps(temp_sum, control); 
						r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
					}

					if(r&(1<<3)) {
						best_ind = filter_ind+0;
						const int control = 3|(3<<2)|(3<<4)|(3<<6);
						max_fil = _mm_permute_ps(temp_sum, control); 
						r=_mm_movemask_ps( _mm_cmp_ps(temp_sum, max_fil, _CMP_GT_OS));
					}
					_mm_store_ps(max_sim, max_fil);

					*ass_out = (float)best_ind;
					*wgt_out = max_sim[0];

					fr_ptr++;
					ass_out++;
					wgt_out++;
				}
			}
	}
	free(pixel_offsets);
}

// *********************************************************************************************************************
// *  FILTER2: AVX2 implementation of histogram computation 
// *********************************************************************************************************************
void block_histogram_AVX(float *ptr_his, float *id_data, float *wt_data, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind) {
    int n_parts_y = (im_height-2) / cell_size;
    int n_parts_x = (im_width-2) / cell_size;
    int start_i = 1;
    int start_j = 1;

    __m256i bins;
	__m256 weights;
    __m256i bins_mask = _mm256_set1_epi32(max_bin - 1);
    __m256 zeros = _mm256_setzero_ps();
    __m256 pitch_his_vec = _mm256_set1_ps(pitch_his);
    __m256 pitch_ind_vec = _mm256_set1_ps(pitch_ind);

    for (int write_i=0; write_i<n_parts_y; write_i++) {
        for (int write_j=0; write_j<n_parts_x; write_j++) {
            int out_ind = (write_i*n_parts_x + write_j) * pitch_his;
            int read_i = (start_i + (write_i * cell_size)) * pitch_ind;

            for (int i=0; i<cell_size; i++) {
                int read_j = start_j + write_j * cell_size ;
                int j = 0;

                for (; j+7<cell_size; j+=8) {
                    bins = _mm256_loadu_si256((__m256i *)(id_data + read_i + read_j + j));
                    bins = _mm256_and_si256(bins, bins_mask);

                    weights = _mm256_load_ps(wt_data + read_i + read_j + j);
                    ptr_his[out_ind + bins[0]] += weights[0];
                    ptr_his[out_ind + bins[1]] += weights[1];
                    ptr_his[out_ind + bins[2]] += weights[2];
                    ptr_his[out_ind + bins[3]] += weights[3];
                    ptr_his[out_ind + bins[4]] += weights[4];
                    ptr_his[out_ind + bins[5]] += weights[5];
                    ptr_his[out_ind + bins[6]] += weights[6];
                    ptr_his[out_ind + bins[7]] += weights[7];
                }

                for (; j<cell_size; j++) {
                    int bin_ind = (int)id_data[read_i + read_j + j] & (max_bin - 1);
                    float weight = wt_data[read_i + read_j + j];
                    ptr_his[out_ind + bin_ind] += weight;
                }
                read_i += pitch_ind;
            }
            out_ind += pitch_his_vec[0];
        }
    }
}

// *********************************************************************************************************************
// *  FILTER 3: AVX2 implementation of the euclidean distance function
// *********************************************************************************************************************
void pwdist_AVX(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth) {
	__m256 m_a, m_b, m_dif, m_square, m_sum, t1, t2, m_sum_permute, tmp;
	__m128 temp;
	float sum;

	for (int i=0; i<aheight; i++){
		for (int j=0; j<bheight; j++){
			m_sum =_mm256_set1_ps(0.0f);
			sum=0.0;
			for (int k = 0; k < awidth; k+=8){
				m_a = _mm256_load_ps (&ptra[0]+i*awidth+k);
				m_b = _mm256_load_ps (&ptrb[0]+j*awidth+k);

				m_dif =  _mm256_sub_ps(m_a,m_b);
				m_square =  _mm256_mul_ps(m_dif, m_dif);
				m_sum =  _mm256_add_ps(m_sum,m_square);
			}
			m_sum_permute = _mm256_permute2f128_ps(m_sum, m_sum, 0x1);
			m_sum = _mm256_hadd_ps(m_sum, m_sum_permute);
			m_sum = _mm256_hadd_ps(m_sum, m_sum);
			m_sum = _mm256_hadd_ps(m_sum, m_sum);
			temp = _mm256_extractf128_ps(m_sum,0);
			sum = _mm_cvtss_f32(temp);
			out_data[i*owidth+j]= sum;
		}
	}		
}

// This function is a modified version of the pwdist_AVX function, divides the matrix into smaller blocks that can fit into the processor cache, minimizing cache misses and improving data locality.
void pwdist_AVX_cache_locality(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth) {
	// Private variables
	__m256 m_a, m_b, m_dif, m_square, m_sum;
	float sum;

    const int simd_width = 8; // AVX2 registers can process 8 floats at a time, AVX-512 can process 16 floats at a time
    const int adatawidth_aligned = (adatawidth + simd_width - 1) / simd_width * simd_width;
    const int block_size = 64; // Adjust this value based on the size of your cache

    for (int ii = 0; ii < aheight; ii += block_size) {
        for (int jj = 0; jj < bheight; jj += block_size) {
			// Within these loops, the original 'i' and 'j' loops now iterate over smaller blocks of the block_size*block_size arrays
            for (int i = ii; i < std::min(ii + block_size, aheight); i++) {
                for (int j = jj; j < std::min(jj + block_size, bheight); j++) {
                    m_sum = _mm256_setzero_ps();
                    for (size_t k = 0; k < adatawidth_aligned; k += simd_width) {
                        m_a = _mm256_load_ps(ptra + i * awidth + k);
                        m_b = _mm256_load_ps(ptrb + j * awidth + k);
                        m_dif = _mm256_sub_ps(m_a, m_b);
                        m_square = _mm256_mul_ps(m_dif, m_dif);
                        m_sum = _mm256_add_ps(m_sum, m_square);
                    }
                    m_sum = _mm256_hadd_ps(m_sum, m_sum);
                    m_sum = _mm256_hadd_ps(m_sum, m_sum);
                    sum = _mm256_cvtss_f32(_mm256_hadd_ps(m_sum, m_sum));
                    out_data[i * owidth + j] = sum;
                }
            }
        }
    }
}