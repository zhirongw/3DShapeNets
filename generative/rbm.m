function [model, hidden_prob] = rbm(model, data_list, data, param)
% RBM training.

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

% Initialization for the parameter b and c.
%data_mean = squeeze(mean(data,1));
%model.layers{l}.b = log( max(0.01,data_mean) ./ max(0.01, 1 - data_mean) );
%if sparse_cost > 0
    %model.layers{l}.c(:) = log( sparse_target ./ ( 1 - sparse_target));
%end
for iter = 1 : param.epochs
    shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = data(batch_index, :);
		
        % positive phase : use data-dependent expectation
        hidden_presigmoid = bsxfun(@plus, ...
            batch * model.layers{l}.w, model.layers{l}.c);
        hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
        hidden_sample = single(hidden_prob > rand(size(hidden_prob)));
		
        pos_grdw = batch' * hidden_prob ./ batch_size;
        pos_grdc = mean(hidden_prob,1);
        pos_grdb = mean(batch, 1);
	
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
                chain * model.layers{l}.w', model.layers{l}.b);
            visible_prob = 1 ./ ( 1 + exp(-visible_presigmoid) );
            
            if persistant
                % get the samples
                visible_sample = single(visible_prob > rand(size(visible_prob)));
            else
                visible_sample = visible_prob;
            end
            
            % propup
            neg_hidden_presigmoid = bsxfun(@plus,...
                visible_sample * model.layers{l}.w, model.layers{l}.c);
            neg_hidden_prob = 1 ./ ( 1 + exp(-neg_hidden_presigmoid) );
            
            chain = single(neg_hidden_prob > rand(size(neg_hidden_prob)));
        end
        if persistant
            persistant_chain = chain;
        end

        neg_grdw = visible_sample' * neg_hidden_prob ./ batch_size;
        neg_grdc = mean(neg_hidden_prob, 1);
        neg_grdb = mean(visible_sample, 1);
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
    end
    % Evaluate the model / compute various cost. 
    if mod(iter,5) == 0
        % get the free energy
        train_energy = free_energy(model, data, l);
        train_energy = train_energy(train_energy ~= Inf);
        fprintf('ITER: %d, FREE-ENERGY, train: %f\n', iter,...
            mean(train_energy));
        
        train_err = get_cross_entropy_all(model, data_list, [], l);
        fprintf('ITER: %d CROSS-ENTROPY: train: %f\n', iter,...
            mean(train_err));
        
        fprintf('hidmeans : %f\n', mean(hidmeans(:)));
        % monitor the weight
        W = model.layers{l}.w; B = model.layers{l}.b; C = model.layers{l}.c;
        dW = model.layers{l}.grdw; dB = model.layers{l}.grdb; dC = model.layers{l}.grdc;
        fprintf('abs mean: W: %f, dW: %f\n',mean(abs(W(:))), mean(abs(dW(:))));
        fprintf('abs mean: B: %f, dB: %f\n',mean(abs(B(:))), mean(abs(dB(:))));
        fprintf('abs mean: C: %f, dC: %f\n',mean(abs(C(:))), mean(abs(dC(:))));
    end
end

% compute the final hidden activations
hidden_presigmoid = bsxfun(@plus, ...
    data * model.layers{l}.w, model.layers{l}.c);
hidden_prob = 1 ./ ( 1 + exp(-hidden_presigmoid));

