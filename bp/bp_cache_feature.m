function bp_cache_feature(model)
% cache CNN features for classification and retrieval.

rng('shuffle');
debug = 0;
kernels;

num_layer = length(model.layers);

phase = {'train', 'test'};

for p = 1 : numel(phase)
    data_size = model.volume_size + 2 * model.pad_size;
    data_list = read_data_list(model.data_path, model.classnames, ....
        data_size, 'train', debug);
    for c = 1 : numel(model.classnames)
        the_class = model.classnames{c};
        
        volume_path = [model.data_path '/' the_class '/' num2str(data_size)];

        this_cat_list = data_list{c};
        batch_size = 32;
        n = length(this_cat_list);
        batch_num = ceil(n / batch_size);

        hidden_activation = zeros([n, model.layers{num_layer-1}.layerSize], 'single');
        for b = 1 : batch_num
            batch_end = min(n, batch_size * b);
            batch_index = batch_size * (b-1) + 1 : batch_end;
            batch = read_batch(model, this_cat_list(batch_index), false);
            hidden_activation(batch_index, :,:,:,:) = propagate_batch(model, batch, num_layer);
        end
        hidden_activation = reshape(hidden_activation, n, []);
        features = [hidden_activation];

        save([volume_path '/' phase{p} '_feature.mat'], 'features', '-v7.3');
        fprintf('writing the %s class\n', the_class);
    end
end