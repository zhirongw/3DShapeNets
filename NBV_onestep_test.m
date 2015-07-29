function acc_eval = NBV_onestep_test(model, off_path, angle_inc)
% Next-Best-View one step test. Evaluate the recognition accuracies for
% different view seletion strategies: entropy-based, reconstruction-based,
% random selection, furthest away.

% off_path: testing off path
% angle_inc: rotation increment of each 3D mesh model(360).

rng('shuffle');
kernels;

classes = model.classnames;

% testing on 100 examples, for 4 methods
numInst = 100;
for c = 1 : numel(classes)
    index = 0;
    test_label = zeros(1);
    fprintf('testing on %s class\n', classes{c});
    category_path = [off_path '/' classes{c} '/test'];
    files = dir(category_path);

    methods = zeros(numInst,4);
    entropy_eval = zeros(numInst,4);
    acc_eval = zeros(numInst,4);
    for i = 1 : length(files)
        if strcmp(files(i).name, '.') || strcmp(files(i).name, '..') || files(i).isdir == 1 || ~strcmp(files(i).name(end-2:end), 'off')
            continue;
        end
        if index == 100
            break;
        end
        filename = [category_path '/' files(i).name];
        index = index + 1;
        test_label(index, 1) = c;

        [~, prediction_next, NBV_v, RC_v, RAND_v, FUR_v] = NBV_onestep(model, filename, angle_inc);

        num_v = size(prediction_next,1);

        entropy = zeros(num_v,1);
        for v = 1 : num_v
            temp = prediction_next(v,:);
            temp = temp(temp>0);
            entropy(v) = -sum(temp .* log(temp));
        end

        entropy_eval(index,1) = entropy(NBV_v);
        entropy_eval(index,2) = entropy(RC_v);
        entropy_eval(index,3) = entropy(RAND_v);
        entropy_eval(index,4) = entropy(FUR_v);

        [~ , predicted_label] = max(prediction_next, [], 2);
        acc_eval(index,1) = double(predicted_label(NBV_v) == c);
        acc_eval(index,2) = double(predicted_label(RC_v) == c);
        acc_eval(index,3) = double(predicted_label(RAND_v) == c);
        acc_eval(index,4) = double(predicted_label(FUR_v) == c);

        methods(index,1) = NBV_v;
        methods(index,2) = RC_v;
        methods(index,3) = RAND_v;
        methods(index,4) = FUR_v;
    end

    save(['res/acc_' classes{c}], ['acc_eval']);
    save(['res/entropy_' classes{c}], ['entropy_eval']);
    save(['res/methods_' classes{c}], ['methods']);
end