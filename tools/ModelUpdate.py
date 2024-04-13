import numpy as np
import torch

# sample_merge_type = 'merge'


class ModelUpdate():
    def __init__(self, num_samples):
        self._num_samples = num_samples
        self.learning_rate = 0.009
        self._distance_matrix = np.ones((self._num_samples, self._num_samples), dtype=np.float32) * np.inf
        self.prior_weights = np.zeros((self._num_samples, 1), dtype=np.float32)

        # find the minimum allowed sample weight. samples are discarded if their weights become lower
        self.minimum_sample_weight = self.learning_rate * (1 - self.learning_rate) ** (2 * self._num_samples)

    def points_distance(self, x_np, y_np):
        # x_np and y_np are both type numpy.

        x = x_np.T
        y = y_np.T

        # num_points_x and num_points_y different or same?
        num_points_x = x.shape[0]
        num_points_y = y.shape[0]

        xx = x.dot(x.T)
        yy = y.dot(y.T)
        zz = x.dot(y.T)

        diag_ind_x = np.arange(0, num_points_x).tolist()
        diag_ind_y = np.arange(0, num_points_y).tolist()

        rx = np.tile(np.expand_dims(xx[diag_ind_x, diag_ind_x], axis=0), (num_points_y, 1))
        ry = np.tile(np.expand_dims(yy[diag_ind_y, diag_ind_y], axis=0), (num_points_x, 1))

        # rx = np.vstack([np.expand_dims(xx[diag_ind_x, diag_ind_x], axis=0) for _ in range(num_points_y)])
        # ry = np.vstack([np.expand_dims(yy[diag_ind_y, diag_ind_y], axis=0) for _ in range(num_points_x)])

        distance = rx.T + ry - 2 * zz

        # distance of size N1*N2.

        return np.sum(distance.min(0)) + np.sum(distance.min(1))

    def _merge_samples(self, sample1_PC, sample2_PC, w1, w2, sample_merge_type):
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if sample_merge_type == 'replace':
            merged_sample_PC = sample1_PC
        elif sample_merge_type == 'merge':
            new_pts_idx1 = np.random.randint(low=0, high=sample1_PC.shape[1], size=np.int_(np.ceil(alpha1 * sample1_PC.shape[1])), dtype=np.int64)
            PC1 = sample1_PC[:, new_pts_idx1]
            new_pts_idx2 = np.random.randint(low=0, high=sample2_PC.shape[1], size=np.int_(np.ceil(alpha2 * sample2_PC.shape[1])), dtype=np.int64)
            PC2 = sample2_PC[:, new_pts_idx2]
            merged_sample_PC = np.concatenate([PC1, PC2], axis=1)
        else:
            raise NotImplementedError
        return merged_sample_PC

    def _update_distance_matrix(self, exist_sample_distance, new_sample_distance, id1, id2, w1, w2):
        """
            update the distance matrix
        """
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if id2 < 0:
            # udpate the gram matrix
            if alpha1 == 0:
                self._distance_matrix[:, id1] = new_sample_distance
                self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
                self._distance_matrix[id1, id1] = np.inf
            elif alpha2 == 0:
                # new sample is discard
                pass
            else:
                # new sample is merge with an existing sample
                self._distance_matrix[:, id1] = new_sample_distance
                self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
                self._distance_matrix[id1, id1] = np.inf
        else:
            if alpha1 == 0 or alpha2 == 0:
                raise("Error!")

            # update the distance matrix
            self._distance_matrix[:, id1] = exist_sample_distance
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = np.inf
            self._distance_matrix[:, id2] = new_sample_distance
            self._distance_matrix[id2, :] = self._distance_matrix[:, id2]
            self._distance_matrix[id2, id2] = np.inf

    def update_sample_space_model(self, samplesf, new_train_sample, num_training_samples):
        new_sample_distance = np.zeros(self._num_samples)
        new_sample_distance[:num_training_samples] = np.array([self.points_distance(i, new_train_sample) for i in samplesf])
        new_sample_distance[num_training_samples:] = np.inf

        merged_sample = []
        new_sample = []
        merged_sample_id = -1
        new_sample_id = -1

        if num_training_samples == self._num_samples:
            min_sample_id = np.argmin(self.prior_weights)
            min_sample_weight = self.prior_weights[min_sample_id]
            if min_sample_weight < self.minimum_sample_weight:
                # if any prior weight is less than the minimum allowed weight
                # replace the sample with the new sample
                # udpate distance matrix and the gram matrix
                exist_sample_distance = np.array([])
                self._update_distance_matrix(exist_sample_distance, new_sample_distance, min_sample_id, -1, 0, 1)

                # normalize the prior weights so that the new sample gets weight as the learning rate
                self.prior_weights[min_sample_id] = 0
                self.prior_weights = self.prior_weights * (1 - self.learning_rate) / np.sum(self.prior_weights)
                self.prior_weights[min_sample_id] = self.learning_rate

                # set the new sample and new sample position in the samplesf
                new_sample_id = min_sample_id
                new_sample = new_train_sample
            else:
                # if no sample has low enough prior weight, then we either merge the new sample with
                # an existing sample, or merge two of the existing samples and insert the new sample
                # in the vacated position
                closest_sample_to_new_sample = np.argmin(new_sample_distance)
                new_sample_min_dist = new_sample_distance[closest_sample_to_new_sample]

                # find the closest pair amongst existing samples
                closest_existing_sample_idx = np.argmin(self._distance_matrix.flatten())
                closest_existing_sample_pair = np.unravel_index(closest_existing_sample_idx, self._distance_matrix.shape)
                existing_samples_min_dist = self._distance_matrix[closest_existing_sample_pair[0], closest_existing_sample_pair[1]]
                closest_existing_sample1, closest_existing_sample2 = closest_existing_sample_pair
                if closest_existing_sample1 == closest_existing_sample2:
                    raise("Score matrix diagnoal filled wrongly")

                if new_sample_min_dist < existing_samples_min_dist:
                    # if the min distance of the new sample to the existing samples is less than the
                    # min distance amongst any of the existing samples, we merge the new sample with
                    # the nearest existing sample

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * (1 - self.learning_rate)

                    # set the position of the merged sample
                    merged_sample_id = closest_sample_to_new_sample

                    # extract the existing sample the merge
                    existing_sample_to_merge = samplesf[merged_sample_id]

                    # merge the new_training_sample with existing sample
                    merged_sample = self._merge_samples(existing_sample_to_merge,
                                                      new_train_sample,
                                                      self.prior_weights[merged_sample_id],
                                                      self.learning_rate,
                                                      'merge')
                    if merged_sample.shape[0] == 0:
                        print('合并样本形状为0')
                        raise NotImplementedError
                    # samplesf[merged_sample_id] = merged_sample
                    new_sample_distance = np.array([self.points_distance(i, merged_sample) for i in samplesf])
                    exist_sample_distance = np.array([])
                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(exist_sample_distance,
                                                new_sample_distance,
                                                merged_sample_id,
                                                -1,
                                                self.prior_weights[merged_sample_id, 0],
                                                self.learning_rate)

                    # udpate the prior weight of the merged sample
                    self.prior_weights[closest_sample_to_new_sample] = self.prior_weights[closest_sample_to_new_sample] + self.learning_rate

                else:
                    # if the min distance amongst any of the existing samples is less than the
                    # min distance of the new sample to the existing samples, we merge the nearest
                    # existing samples and insert the new sample in the vacated position

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * (1 - self.learning_rate)

                    if self.prior_weights[closest_existing_sample2] > self.prior_weights[closest_existing_sample1]:
                        tmp = closest_existing_sample1
                        closest_existing_sample1 = closest_existing_sample2
                        closest_existing_sample2 = tmp

                    sample_to_merge1 = samplesf[closest_existing_sample1]
                    sample_to_merge2 = samplesf[closest_existing_sample2]

                    # merge the existing closest samples
                    merged_sample = self._merge_samples(sample_to_merge1,
                                                      sample_to_merge2,
                                                      self.prior_weights[closest_existing_sample1],
                                                      self.prior_weights[closest_existing_sample2],
                                                      'merge')
                    samplesf[closest_existing_sample1] = merged_sample
                    samplesf[closest_existing_sample2] = new_train_sample

                    exist_sample_distance = np.array([self.points_distance(i, merged_sample) for i in samplesf])
                    new_sample_distance = np.array([self.points_distance(i, new_train_sample) for i in samplesf])
                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(exist_sample_distance,
                                                new_sample_distance,
                                                closest_existing_sample1,
                                                closest_existing_sample2,
                                                self.prior_weights[closest_existing_sample1, 0],
                                                self.prior_weights[closest_existing_sample2, 0])

                    # update prior weights for the merged sample and the new sample
                    self.prior_weights[closest_existing_sample1] = self.prior_weights[closest_existing_sample1] + self.prior_weights[closest_existing_sample2]
                    self.prior_weights[closest_existing_sample2] = self.learning_rate

                    # set the mreged sample position and new sample position
                    merged_sample_id = closest_existing_sample1
                    new_sample_id = closest_existing_sample2

                    new_sample = new_train_sample
        else:
            # if the memory is not full, insert the new sample in the next empty location
            sample_position = num_training_samples

            exist_sample_distance = np.array([])

            # update the distance matrix and the gram matrix
            self._update_distance_matrix(exist_sample_distance, new_sample_distance, sample_position, -1, 0, 1)

            # update the prior weight
            if sample_position == 0:
                self.prior_weights[sample_position] = 1
            else:
                self.prior_weights = self.prior_weights * (1 - self.learning_rate)
                self.prior_weights[sample_position] = self.learning_rate

            new_sample_id = sample_position
            new_sample = new_train_sample

        if abs(1 - np.sum(self.prior_weights)) > 1e-5:
            raise("weights not properly udpated")

        return merged_sample, new_sample, merged_sample_id, new_sample_id


if __name__ == '__main__':
    # initialization some parameters.
    # total_frame = 100
    # num_samples = min(num_samples, total_frame)
    # select_model = GMM(num_samples)
    #
    # current_frame = 36
    #
    # if current_frame == 0:
    #     # init distance matrix
    #     print()
    # else:
    #     for frame in current_frame:
    #         # load current frame's points and bounding box.
    #         # PCs[frame], BBs[frame] = "Load the point and box"
    #
    #         # calculate the distance matrix.
    #         print()
    #
    #         # decide whether to update the distance matrix or not, which refers to the four status.
    #         # including some step:
    #         #   step1: decide which status to choose.
    #         #   step2: if id2 < 0, alpha1 = 0, the new sample replace the old one;
    #         #   step2: if id2 < 0, alpha2 = 0, the new sample will be discarded;
    #         #   step2: if id2 < 0, alpha1 != 0 and alpha2 != 0, the new sample and the old one will be merged;
    #         #   step2: if id2 > 0, id1 and id2 will be merged, the new sample will be set to id2.
    #         #   step3: update the distance matrix.
    #         #   step4: update the sample space model.
    #         #   step5: update the tracking template.
    #         #   Note1: How to choose template: the latest one or the weight one.
    #         #   Note2: normalize the distance matrix or not.
    #         #   Note3: How to solve point dimension to match ECO code.
    #         #   Note4: Add or delete.
    #         print()
    # # add to training code.
    #
    # """
    # merged_sample, new_sample, merged_sample_id, new_sample_id = \
    #     self._gmm.update_sample_space_model(self._samplesf, xlf_proj, self._num_training_samples)
    # """
    # print("")
    import numpy as np
    a = ModelUpdate(10)
    # point_old = np.random.randint(0, 9, (3, 2))
    # point_new = np.random.randint(0, 9, (3, 4))
    # a.points_distance(point_old, point_new)
    samplesf = []
    for i in range(60):
        new_train_sample = np.random.rand(3, 45)
        merged_sample, new_sample, merged_sample_id, new_sample_id = a.update_sample_space_model(samplesf, new_train_sample, len(samplesf))
        print(merged_sample_id, new_sample_id)
        if merged_sample_id == -1:
            if len(samplesf) == 10:
                samplesf[new_sample_id] = new_sample
            else:
                samplesf.append(new_sample)
        else:
            if new_sample_id == -1:
                samplesf[merged_sample_id] = merged_sample
            else:
                samplesf[new_sample_id] = new_sample
                samplesf[merged_sample_id] = merged_sample
