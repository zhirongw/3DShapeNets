function err = get_cross_entropy_all(model, new_list, label, to_layer)
% Reconstruction cost of the whole model from input to the layer to_layer.
% Propagate the data to the layer (to_layer) and propagate it down.
% Examine Its reconstruction error.

global kConv_backward  kConv_backward_c kConv_forward2 kConv_forward_c;

fraction = 5;
batch_size = 32;
num_layer = length(model.layers);

err = 0;
if ~isfield(model.layers{2}, 'uw')
    n = length(new_list);
    batch_num = n / batch_size; assert(batch_num == floor(batch_num));
    batch_num = floor(batch_num / fraction); % evaluate 1/fraction% of data for speed
	shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch_data = read_batch(model, new_list(batch_index), false);
        input_data = batch_data;
        
        for l = 2 : to_layer
            if l == 2
                presigmoid = myConvolve2(kConv_forward2, batch_data, model.layers{l}.w, model.layers{l}.stride, 'forward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
                batch_data = sigmoid(presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
                presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.w, model.layers{l}.stride, 'forward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
                batch_data = sigmoid(presigmoid);
            elseif l < num_layer
                temp_data = reshape(batch_data,batch_size,[]);
                presigmoid = bsxfun(@plus, ...
                    temp_data * model.layers{l}.w, model.layers{l}.c);
                batch_data = sigmoid(presigmoid);
            else
                batch_label = label(batch_index, :);
                temp_w = model.layers{l}.w;
                temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
                batch_data = [batch_label, reshape(batch_data,batch_size,[])];
                presigmoid = bsxfun(@plus, ...
                    batch_data * temp_w, model.layers{l}.c);
                batch_data = sigmoid(presigmoid);
            end
        end

        batch_data = single(batch_data > rand(size(batch_data)));

        for l = to_layer : -1 : 2
            if l == 2
                presigmoid = myConvolve(kConv_backward, batch_data, model.layers{l}.w, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = sigmoid(presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
                presigmoid = myConvolve(kConv_backward_c, batch_data, model.layers{l}.w, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = sigmoid(presigmoid);
            elseif l < num_layer
                presigmoid = bsxfun(@plus, batch_data * model.layers{l}.w', model.layers{l}.b);
                batch_data = sigmoid(presigmoid);
                batch_data = reshape(batch_data, [batch_size, model.layers{l-1}.layerSize]);
            else
                presigmoid = bsxfun(@plus, batch_data * model.layers{l}.w', model.layers{l}.b);
                batch_data = sigmoid(presigmoid);
                batch_data = batch_data(:,model.classes+1:end);
            end
        end
        this_err = input_data - batch_data;
        err = err + sum(this_err(:).^2);           
    end
    err = err / (batch_num * batch_size);
else
    n = length(new_list);
    batch_num = n / batch_size; assert(batch_num == floor(batch_num));
    batch_num = floor(batch_num / fraction); % evaluate 1/fraction% of data for speed
	shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch_data = read_batch(model, new_list(batch_index), false);
        input_data = batch_data;
       
        for l = 2 : to_layer
            if l == 2
                presigmoid = myConvolve2(kConv_forward2, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
                batch_data = sigmoid(presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
                presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
                batch_data = sigmoid(presigmoid);
            elseif l < num_layer
                temp_data = reshape(batch_data,batch_size,[]);
                presigmoid = bsxfun(@plus, ...
                    temp_data * model.layers{l}.uw, model.layers{l}.c);
                batch_data = sigmoid(presigmoid);
            else
                batch_label = label(batch_index, :);
                temp_w = model.layers{l}.w;
                temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
                batch_data = [batch_label, reshape(batch_data,batch_size,[])];
                presigmoid = bsxfun(@plus, ...
                    batch_data * temp_w, model.layers{l}.c);
                batch_data = sigmoid(presigmoid);
            end
        end

        batch_data = single(batch_data > rand(size(batch_data)));

        for l = to_layer : -1 : 2
            if l == 2
                presigmoid = myConvolve(kConv_backward, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = sigmoid(presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
                presigmoid = myConvolve(kConv_backward_c, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = sigmoid(presigmoid);
            elseif l < num_layer
                presigmoid = bsxfun(@plus, batch_data * model.layers{l}.dw', model.layers{l}.b);
                batch_data = sigmoid(presigmoid);
                batch_data = reshape(batch_data, [batch_size, model.layers{l-1}.layerSize]);
            else
                presigmoid = bsxfun(@plus, batch_data * model.layers{l}.w', model.layers{l}.b);
                batch_data = sigmoid(presigmoid);
                batch_data = batch_data(:,model.classes+1:end);
            end
        end
        this_err = input_data - batch_data;
        err = err + sum(this_err(:).^2);           
    end
    err = err / (batch_num * batch_size);
end

function y = sigmoid(x)
    y = 1 ./ ( 1 + exp(-x) );
