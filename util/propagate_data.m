function [hidden_activation, label] = propagate_data(model, data_list, layer)
% propagate the data from the bottom to the layer specified.
% data_list: contains the filenames of the data to be processed.
% hidden_activation: returns the activations of number $layer.
% label: returns the label for the data.

batch_size = 32;
[new_list, label] = balance_data(data_list, batch_size);
n = length(new_list);
batch_num = ceil(n / batch_size);

hidden_activation = zeros([n, model.layers{layer-1}.layerSize], 'single');
for b = 1 : batch_num
    batch_end = min(n, batch_size * b);
    batch_index = batch_size * (b-1) + 1 : batch_end;
    batch = read_batch(model, new_list(batch_index), false);
    hidden_activation(batch_index, :,:,:,:) = propagate_batch(model, batch, layer);
end
hidden_activation = reshape(hidden_activation, n, []);
