function [model] = crbm(model, data_list, param)
% Convolutional RBM training of the third and fourth layer.
% data_list: contains an array of filenames. Returned by balance_data.m
% param: training parameters set in run_pretrain.m

global kConv_backward_c kConv_forward_c kConv_weight_c;

lr = param.lr;
l = param.layer;
wd = param.weight_decay;
persistant = param.persistant;
batch_size = param.batch_size;
sparse_damping = param.sparse_damping;
sparse_target = param.sparse_target;
sparse_cost = param.sparse_cost;

fprintf('Begin pretraining the %1.d th CRBM........\n', l-1);
fprintf('%d filters of size %d with stride %d\n', size(model.layers{l}.w,1), size(model.layers{l}.w,2), model.layers{l}.stride);
fprintf('PCD = %d of %d iterations\n', param.persistant, param.epochs);
fprintf('lr = %f. sparse target: %f\n', param.lr, param.sparse_target);

stride = model.layers{l}.stride;
hidmeans = sparse_target * ones([1, model.layers{l}.layerSize], 'single');
% This persistant chain refers to the hidden state
% Also note that persistant_chain is global
persistant_chain = zeros([batch_size, model.layers{l}.layerSize], 'single');

new_list = balance_data(data_list,batch_size);
n = length(new_list);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));
for iter = 1 : param.epochs    
    shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch = read_batch(model, new_list(batch_index), false);
        batch = propagate_batch(model, batch, l);
        
        %% positive phase : use data-dependent expectation
        % first propagate upwards
        hidden_presigmoid = myConvolve(kConv_forward_c, batch, model.layers{l}.w, stride, 'forward');
        hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
        hidden_prob = sigmoid(hidden_presigmoid);
        hidden_sample = single(hidden_prob > rand(size(hidden_prob)));

        % collect learning signals
        pos_grdb = squeeze(mean(batch,1));
        pos_grdc = squeeze(mean(mean(mean(mean(hidden_prob,1),2),3),4));
        pos_grdw = myConvolve(kConv_weight_c, batch, hidden_prob, stride, 'weight');
        
        % this part for sparsity
        hidmeans = sparse_damping * hidmeans + (1 - sparse_damping) * mean(hidden_prob,1);
        sparsegrads_c = sparse_cost * squeeze(mean(mean(mean(hidmeans - sparse_target, 2),3),4));
        
        %% negative phase: drawing fantasies from the current model using persistant CD
        if persistant
            chain = persistant_chain;
        else
            chain = hidden_sample;
        end

        for k = 1 : param.kPCD
            % PROPDOWN
            visible_presigmoid = myConvolve(kConv_backward_c, chain, model.layers{l}.w, stride, 'backward');
            visible_presigmoid = bsxfun(@plus, visible_presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
            visible_prob = sigmoid(visible_presigmoid);

            if ~persistant
                visible_sample = visible_prob;
            else
                visible_sample = single(visible_prob > rand(size(visible_prob)));
            end

            % PROPUP 
            neg_hidden_presigmoid = myConvolve(kConv_forward_c, visible_sample, model.layers{l}.w, stride, 'forward');
            neg_hidden_presigmoid = bsxfun(@plus, neg_hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
            neg_hidden_prob = sigmoid(neg_hidden_presigmoid);
            
            chain = single(neg_hidden_prob > rand(size(neg_hidden_prob)));
        end
        if persistant
            persistant_chain = chain;
        end
        
        neg_grdb = squeeze(mean(visible_sample,1));
        neg_grdc = squeeze(mean(mean(mean(mean(neg_hidden_prob,1),2),3),4));
        neg_grdw = myConvolve(kConv_weight_c, visible_sample, neg_hidden_prob, stride, 'weight');

        % compute the gradient
        if iter <= floor(param.epochs * 0.25)
            momentum = param.momentum(1);
        else
            momentum = param.momentum(2);
        end
        model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * (pos_grdw - neg_grdw - wd * model.layers{l}.w);
        model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * (pos_grdb - neg_grdb) / model.layers{l}.layerSize(1);
        model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * (pos_grdc - neg_grdc - sparsegrads_c);

        model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
        model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
        model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;

    end
    
    if mod(iter,5) == 0
        % Evaluate the model / compute various cost. 
        train_err = get_cross_entropy_all(model, new_list, [], l);
        fprintf('Iter: %d cross-entropy: train: %f\n', iter,...
            mean(train_err));
        
        % monitor the weight
        W = model.layers{l}.w; B = model.layers{l}.b; C = model.layers{l}.c;
        dW = model.layers{l}.grdw; dB = model.layers{l}.grdb; dC = model.layers{l}.grdc;
        fprintf('abs mean: W: %f, dW: %f\n',mean(abs(W(:))), mean(abs(dW(:))));
        fprintf('abs mean: B: %f, dB: %f\n',mean(abs(B(:))), mean(abs(dB(:))));
        fprintf('abs mean: C: %f, dC: %f\n',mean(abs(C(:))), mean(abs(dC(:))));
        fprintf('hidmeans : %f, sparse_cost: %f\n', mean(hidmeans(:)), sparse_cost);
        t = pos_grdc - neg_grdc;
        fprintf('grad: %f, sparse : %f\n', mean(abs(t(:))), mean(abs(sparsegrads_c(:))));
    end
end


function [y] = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
