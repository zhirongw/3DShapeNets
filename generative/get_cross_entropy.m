function err = get_cross_entropy(model, data, l)
% cross-entropy(reconstruction error, I use L2) of each layer. This is used
% to monitor the pretraining progress.
% data: can be real data(for rbm) or data filelist(crbm).

global kConv_backward  kConv_backward_c kConv_forward2 kConv_forward_c;

fraction = 5;
if l == 2
    n = length(data);
    batch_size = 32; batch_num = n / batch_size;
    assert(batch_num == floor(batch_num));
    stride = model.layers{l}.stride;
    
    batch_num = floor(batch_num / fraction); % evaluate 1/fraction% of data for speed
    shuffle_index = randperm(n);
    err = 0;
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = read_batch(model, data(batch_index), false);
        
        hidden_presigmoid = myConvolve2(kConv_forward2, batch, model.layers{l}.w, stride, 'forward');
        hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        hidden_prob = sigmoid(hidden_presigmoid);
        hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
        
        visible_presigmoid = myConvolve(kConv_backward, hidden_sample, model.layers{l}.w, stride, 'backward');
        visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
        visible_prob = sigmoid(visible_presigmoid);
        
        this_err = batch - visible_prob;
        err = err + sum(this_err(:).^2);
    end
    err = err / (batch_num * batch_size);
elseif strcmp(model.layers{l}.type, 'convolution')
    n = length(data);
    batch_size = 32; batch_num = n / batch_size;
    assert(batch_num == floor(batch_num));
    stride = model.layers{l}.stride;
    
    batch_num = floor(batch_num / fraction); % evaluate 1/fraction% of data for speed
    shuffle_index = randperm(n);
    err = 0;
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = read_batch(model, data(batch_index), false);
        batch = propagate_batch(model, batch, l);
        
        hidden_presigmoid = myConvolve(kConv_forward_c, batch, model.layers{l}.w, stride, 'forward');
        hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        hidden_prob = sigmoid(hidden_presigmoid);
        hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
        
        visible_presigmoid = myConvolve(kConv_backward_c, hidden_sample, model.layers{l}.w, stride, 'backward');
        visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
        visible_prob = sigmoid(visible_presigmoid);
        
        this_err = batch - visible_prob;
        err = err + sum(this_err(:).^2);
    end
    err = err / (batch_num * batch_size);
elseif model.classes > 0 && l == length(model.layers) % for the last layer
    n = size(data,1);
    batch_size = 32; batch_num = n / batch_size;
    err = 0;
    temp_w = model.layers{l}.w;
    temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
    for b = 1 : batch_num
        batch = data((b-1)*batch_size + 1 : b * batch_size,:,:,:,:);
        hidden_presigmoid = bsxfun(@plus, ...
			batch * temp_w, model.layers{l}.c);
		hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
		hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
        
        visible_presigmoid = bsxfun(@plus, ...
					hidden_sample * model.layers{l}.w', model.layers{l}.b);
        visible_prob_post = sigmoid(visible_presigmoid(:,model.classes+1:end));
        temp_exponential = exp(bsxfun(@minus,visible_presigmoid(:,1:model.classes),max(visible_presigmoid(:,1:model.classes),[],2)));
        visible_prob_pre = bsxfun(@rdivide, temp_exponential, sum(temp_exponential,2)); 
        
        data_err = batch(:,model.classes+1:end) - visible_prob_post;
        label_err = batch(:, 1:model.classes) - visible_prob_pre;
        err = err + model.duplicate * sum(label_err(:).^2) + sum(data_err(:).^2);
    end
    err = err / n;
else % for original rbm
    n = size(data,1);
    batch_size = 32; batch_num = n / batch_size;
    err = 0;
    for b = 1 : batch_num
        batch = data((b-1)*batch_size + 1 : b * batch_size,:,:,:,:);
        hidden_presigmoid = bsxfun(@plus, ...
			batch * model.layers{l}.w, model.layers{l}.c);
		hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
		hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
        
        visible_presigmoid = bsxfun(@plus, ...
					hidden_sample * model.layers{l}.w', model.layers{l}.b);
        visible_prob = 1 ./ ( 1 + exp(-visible_presigmoid) );
        
        this_err = batch - visible_prob;
        err = err + sum(this_err(:).^2);
    end
    err = err / n;
end

function y = sigmoid(x)
    y = 1 ./ ( 1 + exp(-x) );
