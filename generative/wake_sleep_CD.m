function [model] = wake_sleep_CD(model, data_list, param)
% Generatively fine-tuning the model using the wake sleep algorithm.
% This is the version that I changed, tuning the weight without untying the
% weights. The top RBM keeps a negative persistant chain.
% data_list: an array of filenames returned by balance_data.m
% param: fine-tuning parameters set in run_finetuning.m

display('generatively fine-tuning the model using wake-sleep CD ........');

global kConv_backward  kConv_backward_c kConv_forward2 kConv_forward_c kConv_weight kConv_weight_c;

lr = param.lr;
batch_size = param.batch_size;
num_layer = length(model.layers);
num_classes = model.classes;

if isfield(model, 'persistant_chain')
    persistant_chain = model.persistant_chain;
else
    persistant_chain = zeros([batch_size, model.layers{num_layer}.layerSize], 'single');
end

[new_list, label] = balance_data(data_list, batch_size);
n = length(new_list);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));

err = get_cross_entropy_all(model, new_list, label, num_layer);
fprintf('original cross_entropy is : %f\n', err);
for iter = 1 : param.epochs
    shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch_data = read_batch(model, new_list(batch_index), false);
		batch_label = label(batch_index, :);
        
		%% wake phase. [BOTTOM-UP PASS] Assuming recognition weight is correct, and update the generative weight.
		wake_activations = cell(num_layer, 1);
		wake_activations{1} = batch_data;
		for l = 2 : num_layer
            if l == 2
	            presigmoid = myConvolve2(kConv_forward2, wake_activations{l-1}, model.layers{l}.w, model.layers{l}.stride, 'forward');
	            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
	            wake_activations{l} = sigmoid(presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
            	presigmoid = myConvolve(kConv_forward_c, wake_activations{l-1}, model.layers{l}.w, model.layers{l}.stride, 'forward');
	            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
	            wake_activations{l} = sigmoid(presigmoid);
            elseif l < num_layer
	        	temp_activations = reshape(wake_activations{l-1},batch_size,[]);
	        	presigmoid = bsxfun(@plus, ...
					temp_activations * model.layers{l}.w, model.layers{l}.c);
	        	wake_activations{l} = sigmoid(presigmoid);
            else
                temp_w = model.layers{l}.w;
                temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
                wake_activations{l-1} = [batch_label, reshape(wake_activations{l-1},batch_size,[])];
	        	presigmoid = bsxfun(@plus, ...
					wake_activations{l-1} * temp_w, model.layers{l}.c);
	        	wake_activations{l} = sigmoid(presigmoid);
            end
		end

		% positive statistics for top rbm
		pos_top_w = wake_activations{num_layer - 1}' * wake_activations{num_layer}./ batch_size;
		pos_top_b = mean(wake_activations{num_layer - 1},1);
		pos_top_c = mean(wake_activations{num_layer},1);

		% perform gibbs sampling in the top rbm
        neg_top_states = single(wake_activations{l} > rand(size(wake_activations{l})));
        if param.persistant
            chain = persistant_chain;
        else
            chain = neg_top_states;
        end
        
        for k = 1 : param.kCD
			% prop down
            presigmoid = bsxfun(@plus, chain * model.layers{num_layer}.w', model.layers{num_layer}.b);
			neg_down_prob_post = sigmoid(presigmoid(:, num_classes+1:end));
			temp_exponential = exp(bsxfun(@minus,presigmoid(:,1:model.classes),max(presigmoid(:,1:model.classes),[],2)));
			neg_down_prob_pre = bsxfun(@rdivide, temp_exponential, sum(temp_exponential,2)); 
			neg_down_prob = [neg_down_prob_pre, neg_down_prob_post];
			
            neg_down_post_states = single(neg_down_prob_post > rand(size(neg_down_prob_post)));
		    %neg_down_pre_rand = rand([size(neg_down_prob_pre,1),1],'single');
            %neg_down_pre_states = zeros(size(neg_down_prob_pre),'single')';
            %kMultinomial.ThreadBlockSize = 256;
            %kMultinomial.GridSize = ceil(size(neg_down_prob_pre,2) / 256);
            %neg_down_pre_states_gpu = feval(kMultinomial, neg_down_pre_states, neg_down_prob_pre', neg_down_pre_rand, size(neg_down_prob_pre,2), size(neg_down_prob_pre,1));
            %neg_down_pre_states = single(gather(neg_down_pre_states_gpu))';
			
            neg_down_states = [neg_down_prob_pre, neg_down_post_states];
			%neg_down_states = [neg_down_pre_states, neg_down_post_states];

			% prop up
			presigmoid = bsxfun(@plus,...
						neg_down_states * temp_w, model.layers{num_layer}.c);
			neg_top_prob = sigmoid(presigmoid);
			chain = single(neg_top_prob > rand(size(neg_top_prob)));
        end
        
        if param.persistant
            persistant_chain = chain;
        end
		% negative statistics for top rbm
		neg_top_w = neg_down_states' * neg_top_prob ./ batch_size;
		neg_top_b = mean(neg_down_states, 1);
		neg_top_c = mean(neg_top_prob, 1);

		%% sleep phase. [TOP-DOWN PASS] Assuming generative weight is correct, and update the recognition weight.
		sleep_activations = cell(num_layer , 1);
		sleep_activations{num_layer} = neg_top_prob;
		sleep_activations{num_layer - 1} = neg_down_prob(:,model.classes+1:end);

		% generating top - down
        for l = num_layer - 1 : -1 : 2
			if l == 2
                presigmoid = myConvolve(kConv_backward, sleep_activations{l}, model.layers{l}.w, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
				sleep_activations{l-1} = sigmoid(presigmoid);
			elseif strcmp(model.layers{l}.type, 'convolution')
				presigmoid = myConvolve(kConv_backward_c, sleep_activations{l}, model.layers{l}.w, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
				sleep_activations{l-1} = sigmoid(presigmoid);
            else
                presigmoid = bsxfun(@plus, sleep_activations{l} * model.layers{l}.w', model.layers{l}.b);
                sleep_activations{l-1} = sigmoid(presigmoid);
                sleep_activations{l-1} = reshape(sleep_activations{l-1}, [batch_size, model.layers{l-1}.layerSize]);
			end
        end

		p_rec = cell(num_layer - 2, 1);
		for l = 1 : num_layer - 2
			if l+1 == 2
				presigmoid = myConvolve2(kConv_forward2, sleep_activations{l}, model.layers{l+1}.w, model.layers{l+1}.stride, 'forward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l+1}.c, [2,3,4,5,1]));
		        p_rec{l} = sigmoid(presigmoid);
			elseif strcmp(model.layers{l+1}.type, 'convolution')
				presigmoid = myConvolve(kConv_forward_c, sleep_activations{l}, model.layers{l+1}.w, model.layers{l+1}.stride, 'forward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l+1}.c, [2,3,4,5,1]));
		        p_rec{l} = sigmoid(presigmoid);
            else
                sleep_activations{l} = reshape(sleep_activations{l}, batch_size, []);
                presigmoid = bsxfun(@plus,...
						sleep_activations{l} * model.layers{l+1}.w, model.layers{l+1}.c);
                p_rec{l} = sigmoid(presigmoid);
			end
		end

		%% update parameters
		% top rbm
        if iter <= floor(param.epochs * 0.25)
            momentum = param.momentum(1);
        else
            momentum = param.momentum(2);
        end
        model.layers{num_layer}.grdw = momentum * model.layers{num_layer}.grdw + lr  * (pos_top_w - neg_top_w);
        model.layers{num_layer}.grdb = momentum * model.layers{num_layer}.grdb + lr  * (pos_top_b - neg_top_b);
        model.layers{num_layer}.grdc = momentum * model.layers{num_layer}.grdc + lr  * (pos_top_c - neg_top_c);
		model.layers{num_layer}.w = model.layers{num_layer}.w + model.layers{num_layer}.grdw;
		model.layers{num_layer}.b = model.layers{num_layer}.b + model.layers{num_layer}.grdb;
		model.layers{num_layer}.c = model.layers{num_layer}.c + model.layers{num_layer}.grdc;

		% generative weight
        for l = 2 : num_layer - 1
			if l == 2
                pos = myConvolve(kConv_weight, wake_activations{l-1}, wake_activations{l}, model.layers{l}.stride, 'weight');
                neg = myConvolve(kConv_weight, sleep_activations{l-1}, p_rec{l-1}, model.layers{l}.stride, 'weight');
                model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * model.layers{l}.layerSize(1) * (pos - neg);
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * squeeze(mean(wake_activations{l-1} - sleep_activations{l-1},1));
                model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * squeeze(mean(mean(mean(mean(wake_activations{l} - p_rec{l-1},1),2),3),4));
				model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
				model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
                model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
			elseif strcmp(model.layers{l}.type, 'convolution')
                pos = myConvolve(kConv_weight_c, wake_activations{l-1}, wake_activations{l}, model.layers{l}.stride, 'weight');
                neg = myConvolve(kConv_weight_c, sleep_activations{l-1}, p_rec{l-1}, model.layers{l}.stride, 'weight');
                model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * model.layers{l}.layerSize(1) * (pos - neg);
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * squeeze(mean(wake_activations{l-1} - sleep_activations{l-1},1));
                model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * squeeze(mean(mean(mean(mean(wake_activations{l} - p_rec{l-1},1),2),3),4));
				model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
				model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
                model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
            elseif l == num_layer - 1
                wake_activations{l-1} = reshape(wake_activations{l-1}, batch_size, []);
                pos = wake_activations{l-1}' * wake_activations{l}(:,num_classes+1:end) ./ batch_size;
                neg = sleep_activations{l-1}' * p_rec{l-1} ./ batch_size;
                model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * (pos - neg);
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * mean(wake_activations{l-1} - sleep_activations{l-1}, 1);
                model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * mean(wake_activations{l}(:,num_classes+1:end) - p_rec{l-1}, 1);
                model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
                model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
                model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
            else
                wake_activations{l-1} = reshape(wake_activations{l-1}, batch_size, []);
                pos = wake_activations{l-1}' * wake_activations{l}./ batch_size;
                neg = sleep_activations{l-1}' * p_rec{l-1} ./ batch_size;
                model.layers{l}.grdw = momentum * model.layers{l}.grdw + lr * (pos - neg);
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * mean(wake_activations{l-1} - sleep_activations{l-1}, 1);
                model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * mean(wake_activations{l} - p_rec{l-1}, 1);
                model.layers{l}.w = model.layers{l}.w + model.layers{l}.grdw;
                model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
                model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
			end
        end
    end
    if mod(iter,5) == 0
        err = get_cross_entropy_all(model, new_list, label, num_layer);
        fprintf('iter : %d, cross_entropy is : %f\n', iter, err);
        for l = 2 : num_layer - 1
            W = model.layers{l}.w; dW = model.layers{l}.grdw;
            fprintf('layer: %d, abs mean: W: %f, dW: %f, \n',l, mean(abs(W(:))), mean(abs(dW(:))));
        end
        W = model.layers{num_layer}.w; dW = model.layers{num_layer}.grdw;
        fprintf('layer: %d, abs mean: W: %f, dW: %f\n', num_layer, mean(abs(W(:))), mean(abs(dW(:))));
    end
end

% clean up the gradients
for l = 2 : num_layer
	model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdb');
    model.layers{l} = rmfield(model.layers{l},'grdc');
end

function y = sigmoid(x)
	y = 1 ./ (1 + exp(-x));