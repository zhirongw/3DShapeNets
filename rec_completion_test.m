function [completed_data, predicted_label, energy] = rec_completion_test(model, test_data, mask, show, param)
% Given incomplete 3D shape(tsdf), recognition and completion test for
% multi-class model. 

% Input the trained model, and some fixed masks, this function perform
% completion and recognition simultaneously. Returns the recognition
% accuracy for the incomplete data(test_data) and the completed_data as
% well as the free energy and predicted_label associated with that
% completion.

if ~exist('param','var');
    param.batch_size = 32;
    param.epochs = 50;
    param.gibbs_iter = 1;
    param.earlyStop = false;
end

addpath voxelization;
addpath 3D;
global kConv_backward  kConv_backward_c kConv_forward2 kConv_forward_c;

if ~isfield(model.layers{2},'uw')
    model = merge_model(model);
end

n = size(test_data,1);
num_layer = length(model.layers);
batch_size = 32;
batch_num = ceil(n / batch_size);

predicted_label = zeros(n, model.classes);
completed_data = initialize_missing(test_data, mask);

ncount = 0;
running_mean = zeros(1, model.classes);

for epoch = 1 : param.epochs
    for b = 1 : batch_num
        batch_end = min(n, b * batch_size);
        batch_data = completed_data((b-1) * batch_size+1 : batch_end, :,:,:,:);
        batch_label = predicted_label((b-1) * batch_size+1 : batch_end, :,:,:,:);
        batch_mask = mask((b-1) * batch_size+1 : batch_end,:,:,:);
        this_size = size(batch_data, 1);
        % propagate/inference bottum up using recognition weight. 
        for l = 2 : num_layer - 1
            if l == 2
                hidden_presigmoid = myConvolve2(kConv_forward2, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
                hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
                batch_hidden_prob = sigmoid(hidden_presigmoid);
            elseif strcmp(model.layers{l}.type, 'convolution')
                hidden_presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
                hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
                batch_hidden_prob = sigmoid(hidden_presigmoid);
            else
                batch_data = reshape(batch_data, this_size, []);
                hidden_presigmoid = bsxfun(@plus, ...
                    batch_data * model.layers{l}.uw, model.layers{l}.c);
                batch_hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
            end
            batch_data = batch_hidden_prob;
        end
        
        batch_data = reshape(batch_data, this_size, []);
        % calculate the free energy for each label hypothesis
        %for c = 1 : model.classes
        %    try_label = zeros(batch_size, model.classes);
        %    try_label(:,c) = 1;
        %    batch_label(:,c) = free_energy(model, [try_label, batch_data], num_layer);
        %end
        
        hn_1 = batch_data;
        hn_1 = single(hn_1 > rand(size(hn_1)));
        
        temp_w = model.layers{num_layer}.w;
        temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
        for i = 1 : param.gibbs_iter
            % alternating gibbs
            % prop up
            hn = bsxfun(@plus, [batch_label, hn_1] * temp_w, model.layers{num_layer}.c);
            hn = 1 ./ (1 + exp(-hn));
            hn = single(hn > rand(size(hn)));

            % prop down
            hn_1 = bsxfun(@plus, hn * model.layers{num_layer}.w', model.layers{num_layer}.b);
            batch_label = exp(bsxfun(@minus, hn_1(:,1:model.classes), max(hn_1(:,1:model.classes), [], 2)));
            batch_label = bsxfun(@rdivide, batch_label, sum(batch_label, 2));
            hn_1 = 1 ./ ( 1 + exp(-hn_1(:,model.classes+1:end)));
        end

        batch_data = reshape(hn_1, [this_size, model.layers{num_layer-1}.layerSize]);
        
        for l = num_layer - 1 : -1 : 2
            if l == 2
                batch_data = reshape(batch_data, [this_size, model.layers{l}.layerSize]);
                presigmoid = myConvolve(kConv_backward, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = 1 ./ ( 1 + exp(-presigmoid));
            elseif strcmp(model.layers{l}.type, 'convolution')
                batch_data = reshape(batch_data, [this_size, model.layers{l}.layerSize]);
                presigmoid = myConvolve(kConv_backward_c, batch_data, model.layers{l}.dw, model.layers{l}.stride, 'backward');
                presigmoid = bsxfun(@plus, presigmoid, permute(model.layers{l}.b, [5,1,2,3,4]));
                batch_data = 1 ./ ( 1 + exp(-presigmoid));
            else
                batch_data = reshape(batch_data, [this_size, model.layers{l}.layerSize]);
                presigmoid = bsxfun(@plus, ...
					batch_data * model.layers{l}.dw', model.layers{l}.b);
                batch_data = 1 ./ ( 1 + exp(-presigmoid) );
            end
        end
        % clamp the real data
        this_data = test_data((b-1) * batch_size + 1 : batch_end,:,:,:);
        batch_data(~batch_mask) = this_data(~batch_mask);
        
        completed_data((b-1) * batch_size + 1 : batch_end,:,:,:) = batch_data;
        predicted_label((b-1) * batch_size + 1 : batch_end,:) = batch_label;
    end
    
    if all(mean(predicted_label,1) == running_mean)
        ncount = ncount + 1;
    else
        running_mean = mean(predicted_label,1);
        ncount = 1;
    end
    % early stop
    if ((~param.earlyStop && epoch >= 100) || (param.earlyStop)) && ncount > 30 || epoch == param.epochs
        break;
    end
end

% calculate the free_energy
energy = zeros(n,1);
for b = 1 : batch_num
    batch_end = min(n, b * batch_size);
    batch_data = completed_data((b-1) * batch_size+1 : batch_end, :,:,:,:);
    batch_label = predicted_label((b-1) * batch_size+1 : batch_end, :,:,:,:);
    this_size = size(batch_data, 1);
    % propagate/inference bottum up using recognition weight. 
    for l = 2 : num_layer - 1
        if l == 2
            hidden_presigmoid = myConvolve2(kConv_forward2, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
            batch_hidden_prob = sigmoid(hidden_presigmoid);
        elseif strcmp(model.layers{l}.type, 'convolution')
            hidden_presigmoid = myConvolve(kConv_forward_c, batch_data, model.layers{l}.uw, model.layers{l}.stride, 'forward');
            hidden_presigmoid = bsxfun(@plus, hidden_presigmoid, permute(model.layers{l}.c, [2,3,4,5,1]));
            batch_hidden_prob = sigmoid(hidden_presigmoid);
        else
            batch_data = reshape(batch_data, this_size, []);
            hidden_presigmoid = bsxfun(@plus, ...
                batch_data * model.layers{l}.uw, model.layers{l}.c);
            batch_hidden_prob = 1 ./ ( 1 + exp(- hidden_presigmoid) );
        end
        batch_data = batch_hidden_prob;
    end
    energy((b-1) * batch_size+1 : batch_end) = free_energy(model, [batch_label, batch_data], num_layer);
end

% show the result
if show
    for i = 1 : n
        the_sample = test_data(i,:,:,:,:);

        figure;
        subplot(1,2,1);
        plot3D(squeeze(completed_data(i,:,:,:,:)) > 0.1);
        p = patch(isosurface(squeeze(completed_data(i,:,:,:)),0.1));
        set(p,'FaceColor','red','EdgeColor','none');
        daspect([1,1,1])
        view(3); axis tight
        camlight 
        lighting gouraud

        subplot(1,2,2);
        plot3D(squeeze(the_sample) > 0);
        p = patch(isosurface(squeeze(the_sample),0.1));
        set(p,'FaceColor','red','EdgeColor','none');
        daspect([1,1,1])
        view(3); axis tight
        camlight 
        lighting gouraud
        pause;
        close(gcf);
    end
end

function data = initialize_missing(data, mask)
    n = size(data,1);
    bin  = 9;
    div = floor(n / bin); rev = mod(n,bin);
    rev_start = 5 - floor(rev / 2); rev_end = 5 + floor((rev-1) / 2);
    if rev == 0
        for i = 1 : bin
            this_data = data((i-1)*div+1:i*div,:,:,:);
            this_mask = mask((i-1)*div+1:i*div,:,:,:);
            this_data(this_mask) = single(rand(size(this_data(this_mask))) > (0.1 * i ));
            data((i-1)*div+1:i*div,:,:,:) = this_data;
        end
    else
        for i = 1 : rev_start-1
            this_data = data((i-1)*div+1:i*div,:,:,:);
            this_mask = mask((i-1)*div+1:i*div,:,:,:);
            this_data(this_mask) = single(rand(size(this_data(this_mask))) > (0.1 * i ));
            data((i-1)*div+1:i*div,:,:,:) = this_data;
        end
        np = (rev_start - 1) * div;
        for i = rev_start : rev_end
            this_data = data(np + (i-rev_start)*(div+1)+1 : np + (i-rev_start+1) * (div+1),:,:,:);
            this_mask = mask(np + (i-rev_start)*(div+1)+1 : np + (i-rev_start+1) * (div+1),:,:,:);
            this_data(this_mask) = single(rand(size(this_data(this_mask))) > 0.1 * i);
            data(np + (i-rev_start)*(div+1)+1 : np + (i-rev_start+1) * (div+1),:,:,:) = this_data;
        end
        np = (rev_start - 1) * div + rev * (div+1);
        for i = rev_end+1 : bin
            this_data = data(np + (i-rev_end-1)*div+1 : np + (i-rev_end) * div,:,:,:);
            this_mask = mask(np + (i-rev_end-1)*div+1 : np + (i-rev_end) * div,:,:,:);
            this_data(this_mask) = single(rand(size(this_data(this_mask))) > 0.1 * i);
            data(np + (i-rev_end-1)*div+1 : np + (i-rev_end) * div,:,:,:) = this_data;
        end
    end
    
function [y] = sigmoid(x)
	y = 1 ./ (1 + exp(-x));
