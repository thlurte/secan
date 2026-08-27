#ifndef SECAN_SEARCH_DISTANCE_H
#define SECAN_SEARCH_DISTANCE_H

float l2_squared(const float *row, const float *query, int dim);
float cosine_distance(const float *row, const float *query, int dim);

float l2_squared_avx2(const float *row, const float *query, int dim);

#endif
