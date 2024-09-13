#include "filters-SIMD.hpp"

void cosine_filter_SIMD(float *fr_data, float *ind, float *val, float *fb_array, const int height, const int width, const int filter_h, const int filter_w, const int n_filters, int pitch) {
    const int apron_y = filter_h / 2;
    const int apron_x = filter_w / 2;
    const int filter_size = filter_h * filter_w;
    const int filter_bank_size = filter_size * n_filters;

    // Changed malloc to vector
    std::vector<int> pixel_offsets(filter_size);

    int oi = 0;
    for (int ii = -apron_y; ii <= apron_y; ii++) {
        for (int jj = -apron_y; jj <= apron_y; jj++) {
            pixel_offsets[oi] = ii * width + jj;
            oi++;
        }
    }

    const int n_threads = 1;
    const int valid_height = height - 2 * apron_y;
    const int height_step = valid_height / n_threads + 1;

    std::array<simd_t, 9> image_cache;
    simd_t temp_sum = 0.0f;
    simd_t max_sim = -1e6f;
    simd_t best_ind = -1;

    for (int tid = 0; tid < n_threads; tid++) {
        const int start_y = apron_y + tid * height_step;
        const int end_y = std::min(start_y + height_step, height - apron_y);
        for (int i = start_y; i < end_y; i++) {
            float *fr_ptr = fr_data + i * width + apron_x;
            float *ass_out = ind + i * pitch / sizeof(float) + apron_x;
            float *wgt_out = val + i * pitch / sizeof(float) + apron_x;

            for (int j = apron_x; j < (width - apron_x); ++j) {
                for (int c = 0; c < 9; ++c) {
                    image_cache[c] = simd_t(&fr_ptr[pixel_offsets[c]], stdx::element_aligned);
                }

                int filter_ind = 0;

                // 96 filters, 9 values each
                while (filter_ind < ((n_filters / 8) * 8)) {
                    temp_sum = 0.0f;

                    for (int c = 0; c < 9; ++c) {
                        simd_t curr_filter{&fb_array[filter_ind * filter_size + c * 8], stdx::element_aligned};
                        temp_sum += image_cache[c] * curr_filter;
                    }

                    stdx::where(temp_sum > max_sim, max_sim) = temp_sum;
                    stdx::where(temp_sum > max_sim, best_ind) = filter_ind / filter_size;

                    filter_ind += 8;
                }

                temp_sum = 0.0f;
                for (int c = 0; c < 9; ++c) {
                    simd_t curr_filter = simd_t{&fb_array[filter_ind * filter_size + c * 4], stdx::element_aligned};
                    temp_sum += image_cache[c] * curr_filter;
                }

                stdx::where(temp_sum > max_sim, max_sim) = temp_sum;
                stdx::where(temp_sum > max_sim, best_ind) = filter_ind / filter_size;

                ass_out[j] = best_ind[0];
                wgt_out[j] = max_sim[0];
                fr_ptr++;
            }
        }
    }
}

// *********************************************************************************************************************
// *  FILTER2: std::experimental::simd implementation of histogram computation
// *********************************************************************************************************************
void block_histogram_SIMD(float *ptr_his, float *id_data, float *wt_data, int max_bin, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind) {
    int n_parts_y = (im_height - 2) / cell_size;
    int n_parts_x = (im_width - 2) / cell_size;
    int start_i = 1;
    int start_j = 1;

    simd_i bins_mask = simd_i(max_bin - 1);
    simd_i bins;
    simd_f weights;

    for (int write_i = 0; write_i < n_parts_y; write_i++) {
        for (int write_j = 0; write_j < n_parts_x; write_j++) {
            int out_ind = (write_i * n_parts_x + write_j) * pitch_his;
            int read_i = (start_i + (write_i * cell_size)) * pitch_ind;

            for (int i = 0; i < cell_size; i++) {
                int read_j = start_j + write_j * cell_size;
                int j = 0;

                for (; j + simd_i::size() - 1 < cell_size; j += simd_i::size()) {
                    simd_i bins{&id_data[read_i + read_j + j], stdx::element_aligned};
                    bins &= bins_mask;
                    simd_f weights{&wt_data[read_i + read_j + j], stdx::element_aligned};
                    for (int k = 0; k < simd_i::size(); k++) {
                        ptr_his[out_ind + bins[k]] += weights[k];
                    }
                }
                // Epilogue
                for (; j < cell_size; j++) {
                    int bin_ind = (int)id_data[read_i + read_j + j] & (max_bin - 1);
                    float weight = wt_data[read_i + read_j + j];
                    ptr_his[out_ind + bin_ind] += weight;
                }
                read_i += pitch_ind;
            }
            out_ind += pitch_his;
        }
    }
}

// *********************************************************************************************************************
// *  FILTER 3: std::experimental::simd implementation of the euclidean distance function
// *********************************************************************************************************************
void pwdist_SIMD(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth) {
    const int simd_width = simd_t::size();
    const int adatawidth_aligned = (adatawidth + simd_width - 1) / simd_width * simd_width;
    const int block_size = 64;

    simd_t m_a, m_b, m_dif, m_sum;
    double sum;

    for (int ii = 0; ii < aheight; ii += block_size) {
        for (int jj = 0; jj < bheight; jj += block_size) {
            for (int i = ii; i < std::min(ii + block_size, aheight); i++) {
                for (int j = jj; j < std::min(jj + block_size, bheight); j++) {
                    m_sum = 0.0f;
                    for (size_t k = 0; k < adatawidth_aligned; k += simd_width) {
                        m_a = simd_t{&ptra[i * awidth + k], stdx::element_aligned};
                        m_b = simd_t{&ptrb[j * awidth + k], stdx::element_aligned};
                        m_dif = m_a - m_b;
                        m_sum += m_dif * m_dif;
                    }
                    sum = stdx::reduce(m_sum, std::plus<>{});
                    out_data[i * owidth + j] = sum;
                }
            }
        }
    }
}
