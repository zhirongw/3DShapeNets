function model = run_finetuning(model)
% run generative finetuning
% Input: generatively pretrained model
% Output: generatively finetuned model

rng('shuffle');
kernels;
debug = 0;

data_list = read_data_list(model.data_path, model.classnames, ...
    model.volume_size + 2 * model.pad_size, 'train', debug);

num_layer = length(model.layers);
for l = 2 : num_layer
    model.layers{l}.grdw = zeros(size(model.layers{l}.w), 'single');
    model.layers{l}.grdb = zeros(size(model.layers{l}.b), 'single');
    model.layers{l}.grdc = zeros(size(model.layers{l}.c), 'single');
end

% fine-tuning phase: wake-sleep algorithm
param = [];
param.epochs = 200;
param.lr = 0.00003;
param.momentum = [0.9, 0.9];
param.kCD = 3;
param.persistant = 1;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;

[model]= wake_sleep(model, data_list, param);
    
save('finetuned_model','model');
