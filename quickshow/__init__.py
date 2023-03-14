# coding=utf-8
# Copyright 2023 parkminwoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# # limitations under the License.


from quickshow.reduce_dim import vis_pca, vis_tsne2d, vis_tsne3d, joint_pca2d, joint_tsne2d
from quickshow.cluster import vis_cluster_plot
from quickshow.eval_clf_model import vis_cm, get_total_cr_df, vis_multi_plot
from quickshow.utils import find_all_files

__all__ = ["vis_tsne2d", "vis_tsne3d", "vis_pca", "vis_cluster_plot", "vis_cm"
           ,"get_total_cr_df", "vis_multi_plot", "find_all_files", "rcparams", "joint_pca2d", "joint_tsne2d"]

__version__ = "0.1.13"