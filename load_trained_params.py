
import h5py


h5 = h5py.File('trained_model.h5', 'r')
data_scaling_weights = list(h5['scalable_data_reuploading']['lambdas:0'])
variational_params = list(h5['scalable_data_reuploading']['thetas:0'])
output_weights = list(h5['Q-values']['obs-weights:0'])