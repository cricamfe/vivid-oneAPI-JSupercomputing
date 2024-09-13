#include "filters-SYCL.hpp"

using namespace std;

/**************************************
 * FILTER 1: GPU
 * ************************************/
// Combinar los dos kernels, para tener alto rendimiento en ambos dispositivos
sycl::event cosine_filter_transpose_sycl(float *frame, float *ind, float *val, float *fb_array_main, const int height, const int width, const int filter_size, const int n_filters, const int f_pitch_f, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
    // Obtener el dispositivo asociado con la cola
    auto device = Q.get_device();
    // Obtener el tamaño máximo de grupo de trabajo soportado por el dispositivo
    auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
    // Ajustar el local_size basado en el tamaño máximo de grupo de trabajo
    const int local_size = std::min(static_cast<int>(std::sqrt(max_work_group_size)), 16);

    auto t_event = Q.submit([&](sycl::handler &h) {
        if (vector_events != nullptr && !vector_events->empty()) {
            // Get the last event in the vector
            h.depends_on(*vector_events);
        }

        sycl::range<2> local_range(local_size, local_size);
        sycl::range<2> global_range((height - 2 + local_size - 1) / local_size * local_size, (width - 2 + local_size - 1) / local_size * local_size);

        if (device.is_gpu()) {
            // Usar local_accessor para GPU
            sycl::local_accessor<float, 1> local_frame(sycl::range<1>(local_size * local_size), h);

            h.parallel_for<>(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
                int local_id_x = item.get_local_id(0);
                int local_id_y = item.get_local_id(1);
                int group_id_x = item.get_group(0);
                int group_id_y = item.get_group(1);
                int local_idx = local_id_y * local_size + local_id_x;

                int posy = group_id_y * local_size + local_id_y;
                int posx = group_id_x * local_size + local_id_x;

                // Cargar datos en memoria local
                if (posy < height && posx < width) {
                    local_frame[local_idx] = frame[posy * f_pitch_f + posx];
                } else {
                    local_frame[local_idx] = 0.0f;
                }

                item.barrier(sycl::access::fence_space::local_space);

                if (posy >= height - 2 || posx >= width - 2)
                    return;

                float img[9];
                img[0] = local_frame[local_id_y * local_size + local_id_x];
                img[1] = local_frame[local_id_y * local_size + local_id_x + 1];
                img[2] = local_frame[local_id_y * local_size + local_id_x + 2];
                img[3] = local_frame[(local_id_y + 1) * local_size + local_id_x];
                img[4] = local_frame[(local_id_y + 1) * local_size + local_id_x + 1];
                img[5] = local_frame[(local_id_y + 1) * local_size + local_id_x + 2];
                img[6] = local_frame[(local_id_y + 2) * local_size + local_id_x];
                img[7] = local_frame[(local_id_y + 2) * local_size + local_id_x + 1];
                img[8] = local_frame[(local_id_y + 2) * local_size + local_id_x + 2];

                float curval = -1e6;
                float curid = -1;

                for (int filter_id = 0; filter_id < n_filters; filter_id++) {
                    float tmpval = 0.0f;
                    int fi = filter_id * filter_size;

                    tmpval += fb_array_main[fi] * img[0];
                    tmpval += fb_array_main[fi + 1] * img[1];
                    tmpval += fb_array_main[fi + 2] * img[2];
                    tmpval += fb_array_main[fi + 3] * img[3];
                    tmpval += fb_array_main[fi + 4] * img[4];
                    tmpval += fb_array_main[fi + 5] * img[5];
                    tmpval += fb_array_main[fi + 6] * img[6];
                    tmpval += fb_array_main[fi + 7] * img[7];
                    tmpval += fb_array_main[fi + 8] * img[8];

                    tmpval = sycl::fabs(tmpval);

                    if (tmpval > curval) {
                        curid = filter_id;
                        curval = tmpval;
                    }
                }

                const int o_pos = (posy + 1) * f_pitch_f + posx + 1;
                ind[o_pos] = curid;
                val[o_pos] = curval;
            });
        } else {
            // Usar el kernel original optimizado para CPU
            h.parallel_for<>(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
                int posy = item.get_global_id(0);
                int posx = item.get_global_id(1);

                if (posy >= height - 2 || posx >= width - 2)
                    return;

                float img0 = frame[posy * f_pitch_f + posx];
                float img1 = frame[posy * f_pitch_f + posx + 1];
                float img2 = frame[posy * f_pitch_f + posx + 2];
                float img3 = frame[(posy + 1) * f_pitch_f + posx];
                float img4 = frame[(posy + 1) * f_pitch_f + posx + 1];
                float img5 = frame[(posy + 1) * f_pitch_f + posx + 2];
                float img6 = frame[(posy + 2) * f_pitch_f + posx];
                float img7 = frame[(posy + 2) * f_pitch_f + posx + 1];
                float img8 = frame[(posy + 2) * f_pitch_f + posx + 2];

                float curval = -1e6;
                float curid = -1;
                int fi = 0;

                for (int filter_id = 0; filter_id < n_filters; filter_id++) {
                    float tmpval = 0.0f;

                    tmpval += fb_array_main[fi++] * img0;
                    tmpval += fb_array_main[fi++] * img1;
                    tmpval += fb_array_main[fi++] * img2;

                    tmpval += fb_array_main[fi++] * img3;
                    tmpval += fb_array_main[fi++] * img4;
                    tmpval += fb_array_main[fi++] * img5;

                    tmpval += fb_array_main[fi++] * img6;
                    tmpval += fb_array_main[fi++] * img7;
                    tmpval += fb_array_main[fi++] * img8;

                    tmpval = sycl::fabs(tmpval);

                    if (tmpval > curval) {
                        curid = filter_id;
                        curval = tmpval;
                    }
                }

                const int o_pos = (posy + 1) * f_pitch_f + posx + 1;
                ind[o_pos] = curid;
                val[o_pos] = curval;
            });
        }
    });

    if (vector_events == nullptr) {
        t_event.wait();
    }

    return t_event;
}

// Optimizado para funcionar bien tanto en GPU como en CPU
// sycl::event cosine_filter_transpose_sycl(float *frame, float *ind, float *val, float *fb_array_main, const int height, const int width, const int filter_size, const int n_filters, const int f_pitch_f, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
//     // Obtener el dispositivo asociado con la cola
//     auto device = Q.get_device();
//     // Obtener el tamaño máximo de grupo de trabajo soportado por el dispositivo
//     auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
//     // Ajustar el local_size basado en el tamaño máximo de grupo de trabajo
//     const int local_size = std::min(static_cast<int>(std::sqrt(max_work_group_size)), 16);

//     auto t_event = Q.submit([&](sycl::handler &h) {
//         if (depends_on != nullptr && !depends_on->empty()) {
//             h.depends_on(*depends_on);
//         }

//         sycl::range<2> local_range(local_size, local_size);
//         sycl::range<2> global_range((height - 2 + local_size - 1) / local_size * local_size, (width - 2 + local_size - 1) / local_size * local_size);

//         h.parallel_for<>(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
//             int posy = item.get_global_id(0);
//             int posx = item.get_global_id(1);

//             if (posy >= height - 2 || posx >= width - 2)
//                 return;

//             float img0 = frame[posy * f_pitch_f + posx];
//             float img1 = frame[posy * f_pitch_f + posx + 1];
//             float img2 = frame[posy * f_pitch_f + posx + 2];
//             float img3 = frame[(posy + 1) * f_pitch_f + posx];
//             float img4 = frame[(posy + 1) * f_pitch_f + posx + 1];
//             float img5 = frame[(posy + 1) * f_pitch_f + posx + 2];
//             float img6 = frame[(posy + 2) * f_pitch_f + posx];
//             float img7 = frame[(posy + 2) * f_pitch_f + posx + 1];
//             float img8 = frame[(posy + 2) * f_pitch_f + posx + 2];

//             float curval = -1e6;
//             float curid = -1;
//             int fi = 0;

//             for (int filter_id = 0; filter_id < n_filters; filter_id++) {
//                 float tmpval = 0.0f;

//                 tmpval += fb_array_main[fi++] * img0;
//                 tmpval += fb_array_main[fi++] * img1;
//                 tmpval += fb_array_main[fi++] * img2;

//                 tmpval += fb_array_main[fi++] * img3;
//                 tmpval += fb_array_main[fi++] * img4;
//                 tmpval += fb_array_main[fi++] * img5;

//                 tmpval += fb_array_main[fi++] * img6;
//                 tmpval += fb_array_main[fi++] * img7;
//                 tmpval += fb_array_main[fi++] * img8;

//                 tmpval = sycl::fabs(tmpval);

//                 if (tmpval > curval) {
//                     curid = filter_id;
//                     curval = tmpval;
//                 }
//             }

//             const int o_pos = (posy + 1) * f_pitch_f + posx + 1;
//             ind[o_pos] = curid;
//             val[o_pos] = curval;
//         });
//     });

//     if (depends_on == nullptr) {
//         t_event.wait();
//     }

//     return t_event;
// }

// Optimizado para dar el máximo rendimiento en GPU, en CPU no funciona bien
// sycl::event cosine_filter_transpose_sycl(float *frame, float *ind, float *val, float *fb_array_main, const int height, const int width, const int filter_size, const int n_filters, const int f_pitch_f, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
//     // Obtener el dispositivo asociado con la cola
//     auto device = Q.get_device();
//     // Obtener el tamaño máximo de grupo de trabajo soportado por el dispositivo
//     auto max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();
//     // Ajustar el local_size basado en el tamaño máximo de grupo de trabajo
//     const int local_size = std::min(static_cast<int>(std::sqrt(max_work_group_size)), 16);

//     auto t_event = Q.submit([&](sycl::handler &h) {
//         if (depends_on != nullptr && !depends_on->empty()) {
//             h.depends_on(*depends_on);
//         }

//         sycl::range<2> local_range(local_size, local_size);
//         sycl::range<2> global_range((height - 2 + local_size - 1) / local_size * local_size, (width - 2 + local_size - 1) / local_size * local_size);

//         // Usar local_accessor en lugar de accessor con target::local
//         sycl::local_accessor<float, 1> local_frame(sycl::range<1>(local_size * local_size), h);

//         h.parallel_for<>(sycl::nd_range<2>(global_range, local_range), [=](sycl::nd_item<2> item) {
//             int local_id_x = item.get_local_id(0);
//             int local_id_y = item.get_local_id(1);
//             int group_id_x = item.get_group(0);
//             int group_id_y = item.get_group(1);
//             int local_idx = local_id_y * local_size + local_id_x;

//             int posy = group_id_y * local_size + local_id_y;
//             int posx = group_id_x * local_size + local_id_x;

//             // Cargar datos en memoria local
//             if (posy < height && posx < width) {
//                 local_frame[local_idx] = frame[posy * f_pitch_f + posx];
//             } else {
//                 local_frame[local_idx] = 0.0f;
//             }

//             item.barrier(sycl::access::fence_space::local_space);

//             if (posy >= height - 2 || posx >= width - 2)
//                 return;

//             float img[9];
//             img[0] = local_frame[local_id_y * local_size + local_id_x];
//             img[1] = local_frame[local_id_y * local_size + local_id_x + 1];
//             img[2] = local_frame[local_id_y * local_size + local_id_x + 2];
//             img[3] = local_frame[(local_id_y + 1) * local_size + local_id_x];
//             img[4] = local_frame[(local_id_y + 1) * local_size + local_id_x + 1];
//             img[5] = local_frame[(local_id_y + 1) * local_size + local_id_x + 2];
//             img[6] = local_frame[(local_id_y + 2) * local_size + local_id_x];
//             img[7] = local_frame[(local_id_y + 2) * local_size + local_id_x + 1];
//             img[8] = local_frame[(local_id_y + 2) * local_size + local_id_x + 2];

//             float curval = -1e6;
//             float curid = -1;

//             for (int filter_id = 0; filter_id < n_filters; filter_id++) {
//                 float tmpval = 0.0f;
//                 int fi = filter_id * filter_size;

//                 tmpval += fb_array_main[fi] * img[0];
//                 tmpval += fb_array_main[fi + 1] * img[1];
//                 tmpval += fb_array_main[fi + 2] * img[2];
//                 tmpval += fb_array_main[fi + 3] * img[3];
//                 tmpval += fb_array_main[fi + 4] * img[4];
//                 tmpval += fb_array_main[fi + 5] * img[5];
//                 tmpval += fb_array_main[fi + 6] * img[6];
//                 tmpval += fb_array_main[fi + 7] * img[7];
//                 tmpval += fb_array_main[fi + 8] * img[8];

//                 tmpval = sycl::fabs(tmpval);

//                 if (tmpval > curval) {
//                     curid = filter_id;
//                     curval = tmpval;
//                 }
//             }

//             const int o_pos = (posy + 1) * f_pitch_f + posx + 1;
//             ind[o_pos] = curid;
//             val[o_pos] = curval;
//         });
//     });

//     if (depends_on == nullptr) {
//         t_event.wait();
//     }

//     return t_event;
// }

// *********************************************************************************************************************
// FILTER 2:
// *********************************************************************************************************************
sycl::event block_histogram_sycl(float *ptr_his, float *ptr_ind, float *ptr_val, int cell_size, int im_height, int im_width, float pitch_his, float pitch_ind, float pitch_val, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
    const int histogram_pitch_f = pitch_his;
    const int assignments_pitch_f = pitch_ind;
    const int weights_pitch_f = pitch_val;

    const int n_parts_y = (im_height - 2) / cell_size;
    const int n_parts_x = 74;

    auto t_event = Q.submit([&](sycl::handler &h) {
        if (vector_events != nullptr && !vector_events->empty()) {
            h.depends_on(*vector_events);
        }
        h.parallel_for<>(sycl::range<2>(n_parts_y, n_parts_x), [=](sycl::id<2> idx) {
            int block_y = idx[0];
            int block_x = idx[1];

            const int pix_y = block_y * cell_size + 1;
            const int pix_x = block_x * cell_size + 1;

            for (int i = 0; i < cell_size; i++) {
                for (int j = 0; j < cell_size; j++) {
                    const float aval = ptr_ind[(pix_y + j) * assignments_pitch_f + pix_x + i];
                    const float wval = ptr_val[(pix_y + j) * weights_pitch_f + pix_x + i];
                    const int block = block_y * n_parts_x + block_x;
                    ptr_his[block * histogram_pitch_f + (int)aval] += wval;
                }
            }
        });
    });
    if (vector_events == nullptr) {
        t_event.wait();
    }

    return t_event;
}

// *********************************************************************************************************************
// FILTER 3:
// *********************************************************************************************************************
sycl::event pwdist_sycl_basic(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int a_datawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
    auto t_event = Q.submit([&](sycl::handler &h) {
        if (vector_events != nullptr && !vector_events->empty()) {
            h.depends_on(*vector_events);
        }
        h.parallel_for<>(sycl::range<2>(aheight, bheight), [=](sycl::id<2> idx) {
            int i = idx[0];
            int j = idx[1];

            float sum = 0.0;

            int posa = i * awidth;
            int posb = j * awidth;

            for (size_t filter_id = 0; filter_id < a_datawidth; filter_id++) {
                float diff = ptra[posa + filter_id] - ptrb[posb + filter_id];
                sum = sycl::mad(diff, diff, sum);
            }

            out_data[i * owidth + j] = sum;
        });
    });
    if (vector_events == nullptr) {
        t_event.wait();
    }

    return t_event;
}

template <size_t tile_size>
sycl::event pwdist_sycl_tiled_float4(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
    auto nd_range = SYCLUtils::generate2DRange(tile_size, aheight, bheight);
    auto global_range = nd_range.get_group_range();
    auto local_range = nd_range.get_local_range();

    auto M = global_range.get(0);
    auto N = global_range.get(1);

    auto t_event = Q.submit([&](sycl::handler &h) {
        if (vector_events != nullptr && !vector_events->empty()) {
            h.depends_on(*vector_events);
        }
        // Create local memory
        sycl::local_accessor<sycl::float4> local_a(tile_size * tile_size / 4, h);
        sycl::local_accessor<sycl::float4> local_b(tile_size * tile_size / 4, h);
        h.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            int i = item.get_global_id(0);
            int j = item.get_global_id(1);

            int row = item.get_local_id(0);
            int col = item.get_local_id(1);

            sycl::float4 sum = sycl::float4(0.0);

            for (int kk = 0; kk < adatawidth; kk += tile_size) {
                if (i < aheight && (col + kk) < adatawidth) {
                    local_a[row * tile_size / 4 + col / 4] = sycl::float4(ptra[i * awidth + (col + kk)], ptra[i * awidth + (col + kk) + 1], ptra[i * awidth + (col + kk) + 2], ptra[i * awidth + (col + kk) + 3]);
                }

                if ((row + kk) < bheight && j < owidth) {
                    local_b[row * tile_size / 4 + col / 4] = sycl::float4(ptrb[(row + kk) * awidth + j], ptrb[(row + kk) * awidth + j + 1], ptrb[(row + kk) * awidth + j + 2], ptrb[(row + kk) * awidth + j + 3]);
                }

                item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int k = 0; k < tile_size; k += 4) {
                    sycl::float4 vec_a, vec_b, diff;
                    if (kk + k < adatawidth) {
                        vec_a = local_a[row * tile_size / 4 + k / 4];
                        vec_b = local_b[col * tile_size / 4 + k / 4];
                        diff = vec_a - vec_b;
                        sum = sycl::mad(diff, diff, sum);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (i < aheight && j < owidth) {
                out_data[i * owidth + j] = sycl::dot(sum, sycl::float4(1.0));
            }
        });
    });

    if (vector_events == nullptr) {
        t_event.wait();
    }

    return t_event;
}

template sycl::event pwdist_sycl_tiled_float4<16>(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events);
template sycl::event pwdist_sycl_tiled_float4<64>(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events);

template <size_t tile_size>
sycl::event pwdist_sycl_tiled(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events) {
    auto nd_range = SYCLUtils::generate2DRange(tile_size, aheight, bheight);
    auto global_range = nd_range.get_group_range();
    auto local_range = nd_range.get_local_range();

    auto M = global_range.get(0);
    auto N = global_range.get(1);

    auto t_event = Q.submit([&](sycl::handler &h) {
        if (vector_events != nullptr && !vector_events->empty()) {
            h.depends_on(*vector_events);
        }
        sycl::local_accessor<float> local_a(tile_size * tile_size, h);
        sycl::local_accessor<float> local_b(tile_size * tile_size, h);

        h.parallel_for(nd_range, [=](sycl::nd_item<2> item) {
            int i = item.get_global_id(0);
            int j = item.get_global_id(1);

            int row = item.get_local_id(0);
            int col = item.get_local_id(1);

            float sum = 0.0;

            for (int kk = 0; kk < adatawidth; kk += tile_size) {
                if (i < aheight && (col + kk) < adatawidth) {
                    local_a[row * tile_size + col] = ptra[i * awidth + (col + kk)];
                }

                if ((row + kk) < bheight && j < owidth) {
                    local_b[row * tile_size + col] = ptrb[(row + kk) * awidth + j];
                }

                item.barrier(sycl::access::fence_space::local_space);
#pragma unroll
                for (int k = 0; k < tile_size; ++k) {
                    int idx = kk + k;
                    float vec_a, vec_b, diff;
                    if (idx < adatawidth) {
                        vec_a = local_a[row * tile_size + k];
                        vec_b = local_b[col * tile_size + k];
                        diff = vec_a - vec_b;
                        sum = sycl::mad(diff, diff, sum);
                    }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (i < aheight && j < owidth) {
                out_data[i * owidth + j] = sum;
            }
        });
    });
    if (vector_events == nullptr) {
        t_event.wait();
    }

    return t_event;
}

template sycl::event pwdist_sycl_tiled<16>(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events);
template sycl::event pwdist_sycl_tiled<64>(float *ptra, float *ptrb, float *out_data, int owidth, int aheight, int awidth, int bheight, int adatawidth, sycl::queue &Q, const std::vector<sycl::event> *vector_events);
