function model = run_pretrain()
% Layer-wise pretraining.

rng('shuffle');
kernels;

param = [];
param.debug = 0;
param.volume_size = 24;
param.pad_size = 3;
data_size = param.volume_size + 2 * param.pad_size;

% data path
param.data_path = './volumetric_data';
%param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'};
param.classnames = {'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet', ...
           'airplane', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'cone', 'cup', 'curtain', 'door', ...
           'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'person', 'piano', 'plant', ...
           'radio', 'range_hood', 'sink', 'stairs', 'stool', 'tent', 'tv_stand', 'vase', 'wardrobe', 'xbox'};
param.classes = length(param.classnames);
data_list = read_data_list(param.data_path, param.classnames, data_size, 'train', param.debug);

param.network = {
    struct('type', 'input');
    struct('type', 'convolution', 'outputMaps', 48, 'kernelSize', 6, 'actFun', 'sigmoid', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 160, 'kernelSize', 5, 'actFun', 'sigmoid', 'stride', 2);
    struct('type', 'convolution', 'outputMaps', 512, 'kernelSize', 4, 'actFun', 'sigmoid', 'stride', 1);
    struct('type', 'fullconnected', 'size', 1200, 'actFun', 'sigmoid');
    struct('type', 'fullconnected', 'size', 4000, 'actFun', 'sigmoid');
};

% This is to duplicate the labels for the final RBM in order to enforce the
% label training.
param.duplicate = 10;
param.validation = 1;
param.data_size = [data_size, data_size, data_size, 1];

model = initialize_cdbn(param);

fprintf('\nmodel initialzation completed!\n\n');
param = [];
param.layer = 2;
param.epochs = 15;
param.lr = 0.01;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0.01;
param.sparse_cost = 0.001;
[model] = crbm2(model, data_list, param);
%save('model30_l2','model');

param = [];
param.layer = 3;
param.epochs = 30;
param.lr = 0.01;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0.02;
param.sparse_cost = 0.002;
[model] = crbm(model, data_list, param);
%save('model30_l3','model');

param = [];
param.layer = 4;
param.epochs = 50;
param.lr = 0.01;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
[model] = crbm(model, data_list, param);
%save('model30_l4','model');

[hidden_prob_h4, train_label] = propagate_data(model, data_list, 5);

param = [];
param.layer = 5;
param.epochs = 80;
param.lr = 0.003;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.9];
param.kPCD = 1;
param.persistant = 0;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
new_list = balance_data(data_list,param.batch_size);
hidden_prob_h4 = reshape(hidden_prob_h4, size(hidden_prob_h4,1),[]);
[model, hidden_prob_h5] = rbm(model, new_list, hidden_prob_h4, param);
%save('model30_l5','model');

param = [];
param.layer = 6;
param.epochs = 150;
param.lr = 0.003;
param.weight_decay = 1e-5;
param.momentum = [0.5, 0.5];
param.kPCD = 1;
param.persistant = 1;
param.batch_size = 32;
param.sparse_damping = 0;
param.sparse_target = 0;
param.sparse_cost = 0;
new_list = balance_data(data_list,param.batch_size);
[model] = rbm_last(model, new_list, [train_label, hidden_prob_h5], param);

for l = 2 : length(model.layers)
    model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdb');
    model.layers{l} = rmfield(model.layers{l},'grdc');
end

save('pretrained_model','model');
