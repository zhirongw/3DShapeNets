function [model]= wake_sleep(model, data_list, param)
% Generatively fine-tuning the model using the wake sleep algorithm.
% This is the original wake_sleep algorithm, untying the generative weights
% and recognition weights
% data_list: an array of filenames returned by balance_data.m
% param: fine-tuning parameters set in run_finetuning.m
display('generatively fine-tuning the model using wake sleep........');

global kConv_backward kConv_backward_c kConv_forward2 kConv_forward_c kConv_weight kConv_weight_c;

% for original wake-sleep, top-down and bottom-up weights are untied.
if ~isfield(model.layers{2},'uw')
    model = merge_model(model);
end

lr = param.lr;
batch_size = param.batch_size;
num_layer = length(model.layers);
num_classes = model.classes;

persistant_chain = zeros([batch_size, model.layers{num_layer}.layerSize], 'single');

[new_list, label] = balance_data(data_list, batch_size);
n = length(new_list);
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));
for iter = 1 : param.epochs
	shuffle_index = randperm(n);
    for b = 1 : batch_num
        batch_index = shuffle_index((b-1)*batch_size + 1 : b * batch_size);
        batch_data = read_batch(model, new_list(batch_index), false);
		batch_label = label(batch_index, :);
		%% wake phase. [BOTTOM-UP PASS] Assuming recognition weight is correct, and update the generative weight.
		wake_activations = cell(num_layer, 1);
		wake_states = cell(num_layer, 1);
		wake_states{1} = batch_data;
		for l = 2 : num_layer
            if l == 2
	            presigmoid = myConvolve2(kConv_forward2, wake_states{l-1}, model.layers{l}.uw, model.layers{l}.stride, 'forward');
	            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
	            wake_activations{l} = sigmoid(presigmoid);
	            wake_states{l} = single(wake_activations{l} > rand(size(wake_activations{l})));
            elseif strcmp(model.layers{l}.type, 'convolution')
            	presigmoid = myConvolve(kConv_forward_c, wake_states{l-1}, model.layers{l}.uw, model.layers{l}.stride, 'forward');
	            presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.c,[2,3,4,5,1]));
	            wake_activations{l} = sigmoid(presigmoid);
	            wake_states{l} = single(wake_activations{l} > rand(size(wake_activations{l})));
            elseif l < num_layer
	        	wake_states{l-1} = reshape(wake_states{l-1},batch_size,[]);
	        	presigmoid = bsxfun(@plus, ...
					wake_states{l-1} * model.layers{l}.uw, model.layers{l}.c);
	        	wake_activations{l} = sigmoid(presigmoid);
	        	wake_states{l} = single(wake_activations{l} > rand(size(wake_activations{l})));
            else
                temp_w = model.layers{l}.w;
                temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
                wake_states{l-1} = [batch_label, reshape(wake_states{l-1},batch_size,[])];
	        	presigmoid = bsxfun(@plus, ...
					wake_states{l-1} * temp_w, model.layers{l}.c);
	        	wake_activations{l} = sigmoid(presigmoid);
	        	wake_states{l} = single(wake_activations{l} > rand(size(wake_activations{l})));
            end
		end

		% positive statistics for top rbm
		pos_top_w = wake_states{num_layer - 1}' * wake_states{num_layer}./ batch_size;
		pos_top_b = mean(wake_states{num_layer - 1},1);
		pos_top_c = mean(wake_states{num_layer},1);

		% perform gibbs sampling in the top rbm
        if param.persistant
            chain = persistant_chain;
        else
            chain = wake_states{num_layer};
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
		sleep_states = cell(num_layer, 1);
		sleep_states{num_layer} = chain;
		sleep_states{num_layer-1} = neg_down_states(:,model.classes+1:end);

		% generating top - down
        for l = num_layer - 1 : -1 : 2
			if l == 2
                presigmoid = myConvolve(kConv_backward, sleep_states{l}, model.layers{l}.dw, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
				sleep_activations{l-1} = sigmoid(presigmoid);
				%sleep_states{l-1} = single(sleep_activations{l-1} > rand(size(sleep_activations{l-1})));
                sleep_states{l-1} = sleep_activations{l-1};
			elseif strcmp(model.layers{l}.type, 'convolution')
				presigmoid = myConvolve(kConv_backward_c, sleep_states{l}, model.layers{l}.dw, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
				sleep_activations{l-1} = sigmoid(presigmoid);
				sleep_states{l-1} = single(sleep_activations{l-1} > rand(size(sleep_activations{l-1})));
            else
                presigmoid = bsxfun(@plus, sleep_states{l} * model.layers{l}.dw', model.layers{l}.b);
                sleep_activations{l-1} = sigmoid(presigmoid);
                sleep_states{l-1} = single(sleep_activations{l-1} > rand(size(sleep_activations{l-1})));
                sleep_states{l-1} = reshape(sleep_states{l-1}, [batch_size, model.layers{l-1}.layerSize]);
			end
        end
        
		% prediction
		p_gen = cell(num_layer - 2, 1);
        wake_states{num_layer - 1} = wake_states{num_layer - 1}(:, model.classes+1:end);
		for l = 2 : num_layer - 1
			if l == 2
				presigmoid = myConvolve(kConv_backward, wake_states{l}, model.layers{l}.dw, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
		        p_gen{l-1} = sigmoid(presigmoid);
			elseif strcmp(model.layers{l}.type, 'convolution')
                wake_states{l} = reshape(wake_states{l}, [batch_size, model.layers{l}.layerSize]);
				presigmoid = myConvolve(kConv_backward_c, wake_states{l}, model.layers{l}.dw, model.layers{l}.stride, 'backward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
		        p_gen{l-1} = sigmoid(presigmoid);
            else
                presigmoid = bsxfun(@plus, wake_states{l} * model.layers{l}.dw', model.layers{l}.b);
                p_gen{l-1} = sigmoid(presigmoid);
			end
		end

		p_rec = cell(num_layer - 2, 1);
		for l = 1 : num_layer - 2
			if l+1 == 2
				presigmoid = myConvolve2(kConv_forward2, sleep_states{l}, model.layers{l+1}.uw, model.layers{l+1}.stride, 'forward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l+1}.c, [2,3,4,5,1]));
		        p_rec{l} = sigmoid(presigmoid);
			elseif strcmp(model.layers{l+1}.type, 'convolution')
				presigmoid = myConvolve(kConv_forward_c, sleep_states{l}, model.layers{l+1}.uw, model.layers{l+1}.stride, 'forward');
		        presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l+1}.c, [2,3,4,5,1]));
		        p_rec{l} = sigmoid(presigmoid);
            else
                sleep_states{l} = reshape(sleep_states{l}, batch_size, []);
                presigmoid = bsxfun(@plus,...
						sleep_states{l} * model.layers{l+1}.uw, model.layers{l+1}.c);
                p_rec{l} = sigmoid(presigmoid);
			end
		end

		%% update parameters
        if iter <= floor(param.epochs * 0.25)
            momentum = param.momentum(1);
        else
            momentum = param.momentum(2);
        end
		% top rbm
        model.layers{num_layer}.grdw = momentum * model.layers{num_layer}.grdw + lr  * (pos_top_w - neg_top_w);
        model.layers{num_layer}.grdb = momentum * model.layers{num_layer}.grdb + lr  * (pos_top_b - neg_top_b);
        model.layers{num_layer}.grdc = momentum * model.layers{num_layer}.grdc + lr  * (pos_top_c - neg_top_c);
		model.layers{num_layer}.w = model.layers{num_layer}.w + model.layers{num_layer}.grdw;
		model.layers{num_layer}.b = model.layers{num_layer}.b + model.layers{num_layer}.grdb;
		model.layers{num_layer}.c = model.layers{num_layer}.c + model.layers{num_layer}.grdc;

		% generative weight
		for l = 2 : num_layer - 1
			if l == 2
                model.layers{l}.grddw = momentum * model.layers{l}.grddw + lr * prod(model.layers{l}.layerSize(1:3)) * myConvolve(kConv_weight, wake_states{l-1} - p_gen{l-1}, wake_states{l}, model.layers{l}.stride, 'weight');
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * squeeze(mean(wake_states{l-1} - p_gen{l-1},1));
				model.layers{l}.dw = model.layers{l}.dw + model.layers{l}.grddw;
				model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
			elseif strcmp(model.layers{l}.type, 'convolution')
                model.layers{l}.grddw = momentum * model.layers{l}.grddw + lr * prod(model.layers{l}.layerSize(1:3)) * myConvolve(kConv_weight_c, wake_states{l-1} - p_gen{l-1}, wake_states{l}, model.layers{l}.stride, 'weight');
				model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * squeeze(mean(wake_states{l-1} - p_gen{l-1},1));
                model.layers{l}.dw = model.layers{l}.dw + model.layers{l}.grddw;
				model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
            else
                wake_states{l-1} = reshape(wake_states{l-1}, batch_size, []);
                p_gen{l-1} = reshape(p_gen{l-1}, batch_size, []);
                model.layers{l}.grddw = momentum * model.layers{l}.grddw + lr * (wake_states{l-1} - p_gen{l-1})' * wake_states{l} ./ batch_size;
                model.layers{l}.grdb = momentum * model.layers{l}.grdb + lr * mean(wake_states{l-1} - p_gen{l-1}, 1);
                model.layers{l}.dw = model.layers{l}.dw + model.layers{l}.grddw;
                model.layers{l}.b = model.layers{l}.b + model.layers{l}.grdb;
			end
		end
		% recognition weight
		for l = 2 : num_layer - 1
			if l == 2
                model.layers{l}.grduw = momentum * model.layers{l}.grduw + lr * prod(model.layers{l}.layerSize(1:3)) * myConvolve(kConv_weight, sleep_activations{l-1}, sleep_states{l} - p_rec{l-1}, model.layers{l}.stride, 'weight');
				model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * squeeze(mean(sum(sum(sum(sleep_states{l} - p_rec{l-1},1),2),3),4));
                model.layers{l}.uw = model.layers{l}.uw + model.layers{l}.grduw;
				model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
			elseif strcmp(model.layers{l}.type, 'convolution')
                sleep_states{l} = reshape(sleep_states{l}, [batch_size, model.layers{l}.layerSize]);
                model.layers{l}.grduw = momentum * model.layers{l}.grduw + lr * prod(model.layers{l}.layerSize(1:3)) * myConvolve(kConv_weight_c, sleep_states{l-1}, sleep_states{l} - p_rec{l-1}, model.layers{l}.stride, 'weight');
				model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * squeeze(mean(sum(sum(sum(sleep_states{l} - p_rec{l-1},1),2),3),4));
                model.layers{l}.uw = model.layers{l}.uw + model.layers{l}.grduw;
				model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
            else
                sleep_states{l-1} = reshape(sleep_states{l-1}, batch_size, []);
                model.layers{l}.grduw = momentum * model.layers{l}.grduw + lr * sleep_states{l-1}' * (sleep_states{l} - p_rec{l-1}) ./ batch_size;
                model.layers{l}.grdc = momentum * model.layers{l}.grdc + lr * mean(sleep_states{l} - p_rec{l-1}, 1);
                model.layers{l}.uw = model.layers{l}.uw + model.layers{l}.grduw;
                model.layers{l}.c = model.layers{l}.c + model.layers{l}.grdc;
			end
		end
    end
    if mod(iter,5) == 0
        err = get_cross_entropy_all(model, new_list, label, num_layer);
        fprintf('iter : %d, cross_entropy is : %f\n', iter, err);
        for l = 2 : num_layer - 1
            dW = model.layers{l}.dw; ddW = model.layers{l}.grddw;
            uW = model.layers{l}.uw; duW = model.layers{l}.grduw;
            fprintf('layer: %d, abs mean: dW: %f, ddW: %f, uW: %f, duW: %f\n',l, mean(abs(dW(:))), mean(abs(ddW(:))),mean(abs(uW(:))), mean(abs(duW(:))));
        end
        W = model.layers{num_layer}.w; dW = model.layers{num_layer}.grdw;
        fprintf('layer: %d, abs mean: W: %f, dW: %f\n', num_layer, mean(abs(W(:))), mean(abs(dW(:))));
    end
end

% clean up the gradients
for l = 2 : num_layer - 1
	model.layers{l} = rmfield(model.layers{l},'grddw');
	model.layers{l} = rmfield(model.layers{l},'grduw');
    model.layers{l} = rmfield(model.layers{l},'grdb');
    model.layers{l} = rmfield(model.layers{l},'grdc');
end
model.layers{num_layer} = rmfield(model.layers{num_layer}, 'grdw');
model.layers{num_layer} = rmfield(model.layers{num_layer}, 'grdb');
model.layers{num_layer} = rmfield(model.layers{num_layer}, 'grdc');

function y = sigmoid(x)
	y = 1 ./ (1 + exp(-x));