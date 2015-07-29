function [model, hidden_prob] = rbm_last(model, data_list, data, param)
% FPCD RBM training for the joint training of data + label(last top layer).

l = param.layer;
lr = param.lr;
batch_size = param.batch_size;
wd = param.weight_decay;
persistant = param.persistant;
sparse_damping = param.sparse_damping;
sparse_target = param.sparse_target;
sparse_cost = param.sparse_cost;

n = size(data,1);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));

fprintf('Begin pretraining the %1.d th RBM........\n', l-1);
fprintf('layer size: %d\n', size(model.layers{l}.w,1));
fprintf('PCD = %d of %d iterations\n', param.persistant, param.epochs);
fprintf('lr = %f. sparse target: %f\n', param.lr, param.sparse_target);

hidmeans = sparse_target * ones([1, model.layers{l}.layerSize], 'single');
% This persistant chain refers to the hidden state
% Also note that persistant_chain is global
persistant_chain = zeros([batch_size, model.layers{l}.layerSize], 'single');

% initialization of parameter b and c.
%data_mean = squeeze(mean(data,1));
%model.layers{l}.b = log( max(0.01,data_mean) ./ max(0.01, 1 - data_mean) );
%if sparse_cost > 0
    %model.layers{l}.c(:) = log( sparse_target ./ ( 1 - sparse_target));
%end

fast_w = zeros(size(model.layers{l}.w)); fast_grdw = zeros(size(fast_w));
fast_b = zeros(size(model.layers{l}.b)); fast_grdb = zeros(size(fast_b));
fast_c = zeros(size(model.layers{l}.c)); fast_grdc = zeros(size(fast_c));
for iter = 1 : param.epochs
    shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = data(batch_index, :);
		
        % positive phase : use data-dependent expectation
        temp_w = model.layers{l}.w;
        temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
        temp_fast_w = fast_w;
        temp_fast_w(1:model.classes,:) = temp_fast_w(1:model.classes,:);
            hidden_presigmoid = bsxfun(@plus, ...
        batch * temp_w, model.layers{l}.c);
        hidden_prob = sigmoid(hidden_presigmoid);
        hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
		
        pos_grdw = batch' * hidden_prob./ batch_size;
        pos_grdc = mean(hidden_prob,1);
        pos_grdb = mean(batch,1);

        % this part for sparsity
        hidmeans = sparse_damping * hidmeans + (1 - sparse_damping) * mean(hidden_prob,1);
        sparsegrads_c = sparse_cost * (hidmeans - sparse_target);
		
        % negative phase: drawing fantasies from the current model using persistant CD
        if persistant
            chain = persistant_chain;
        else
            chain = hidden_sample;
        end
        
        for k = 1 : param.kPCD
            % propdown
            visible_presigmoid = bsxfun(@plus, ...
                chain * (model.layers{l}.w' + fast_w'), (model.layers{l}.b + fast_b));
            visible_prob_post = sigmoid(visible_presigmoid(:, model.classes+1:end));
            temp_exponential = exp(bsxfun(@minus,visible_presigmoid(:, 1:model.classes),max(visible_presigmoid(:, 1:model.classes),[],2)));
            visible_prob_pre = bsxfun(@rdivide, temp_exponential, sum(temp_exponential,2)); 
            visible_prob = [visible_prob_pre, visible_prob_post];
            
            visible_post_sample = single(visible_prob_post > rand(size(visible_prob_post)));
            visible_pre_sample = mnrnd(1,visible_prob_pre);
            visible_sample = [visible_pre_sample, visible_post_sample];
            
            if ~persistant
                visible_sample = visible_prob;
            end
            
            % propup
            hidden_presigmoid = bsxfun(@plus,...
                visible_sample * (temp_w + temp_fast_w), (model.layers{l}.c + fast_c));
            hidden_prob = sigmoid(hidden_presigmoid);
           
            chain = single(hidden_prob > rand(size(hidden_prob)));
        end
        if persistant
            persistant_chain = chain;
        end
        
        neg_grdw = visible_sample' * hidden_prob ./ batch_size;
        neg_grdb = mean(visible_sample,1);
        neg_grdc = mean(hidden_prob,1);
		
        % compute the gradient
        if iter <= floor(param.epochs * 0.25)
            momentum = param.momentum(1);
        else
            momentum = param.momentum(2);
        end
        model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * (pos_grdw - neg_grdw - wd * model.layers{l}.w);
        model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * (pos_grdb - neg_grdb);
        model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * (pos_grdc - neg_grdc - sparsegrads_c);

        model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
        model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
        model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
        
        fast_grdw = momentum * fast_grdw + lr * (pos_grdw - neg_grdw - wd * model.layers{l}.w);
        fast_grdb = momentum * fast_grdb + lr * (pos_grdb - neg_grdb);
        fast_grdc = momentum * fast_grdc + lr * (pos_grdc - neg_grdc - sparsegrads_c);
        
        fast_w = 19 / 20 * fast_w + fast_grdw;
        fast_b = 19 / 20 * fast_b + fast_grdb;
        fast_c = 19 / 20 * fast_c + fast_grdc;
    end
    
    if iter == floor(0.75 * param.epochs)
        lr = lr / 2;
    end
    if mod(iter, 5) == 0
        % Evaluate the model / compute various cost. 
        train_energy = free_energy(model, data, l);
        train_energy = train_energy(train_energy ~= Inf);
        fprintf('ITER: %d, FREE-ENERGY: train: %f\n', iter,...
            mean(train_energy));
        
        train_err = get_cross_entropy_all(model, data_list, data(:,1:model.classes), l);
        fprintf('ITER: %d CROSS-ENTROPY: train: %f\n', iter,...
            mean(train_err));
        
        fprintf('hidmeans : %f\n', mean(hidmeans(:)));
        % monitor the weight
        W = model.layers{l}.w; B = model.layers{l}.b; C = model.layers{l}.c;
        dW = model.layers{l}.grdw; dB = model.layers{l}.grdb; dC = model.layers{l}.grdc;
        fprintf('abs mean: W: %f, dW: %f\n',mean(abs(W(:))), mean(abs(dW(:))));
        fprintf('abs mean: B: %f, dB: %f\n',mean(abs(B(:))), mean(abs(dB(:))));
        fprintf('abs mean: C: %f, dC: %f\n',mean(abs(C(:))), mean(abs(dC(:))));
        fprintf('hidmeans: %f\n', mean(hidmeans(:)));
    end
end

model.persistant_chain = persistant_chain;

% Compute the final hidden data
hidden_presigmoid = bsxfun(@plus, ...
    data * model.layers{l}.w, model.layers{l}.c);
hidden_prob = sigmoid(hidden_presigmoid);

function y = sigmoid(x)
    y = 1 ./ ( 1 + exp(-x));
