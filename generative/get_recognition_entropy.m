function [entropy, samples, predicted_label]= get_recognition_entropy(model, tsdf_data, param)
% recognition entropy of the incomplete TSDF data.
% tsdf_data: contains surface voxels(+1), empty space(0), and unobserved
% space(-1).
% param: testing parameters. (sampling epochs, number of particles,
% earlyStop or not).
% predicted_label: p(y|tsdf).
% entropy: entropy of predicted_label.
% samples: completed samples of tsdf.

if ~exist('param','var')
    param.nparticles = 32;
    param.epochs = 20;
    param.earlyStop = true;
end

nparticles = param.nparticles;

sample_param = [];
sample_param.epochs = param.epochs;
sample_param.nparticles = nparticles;
sample_param.gibbs_iter = 1;
sample_param.earlyStop = param.earlyStop;

batch_data = repmat(permute(tsdf_data,[4,1,2,3]),nparticles,1); % running n chains altogether
mask = batch_data < 0;

[samples, predicted_label] = rec_completion_test(model, batch_data, mask, 0, sample_param);

% evaluate the entropy
predicted_label = mean(predicted_label,1);
temp = predicted_label(predicted_label>0);
entropy = -sum(temp .* log(temp));
