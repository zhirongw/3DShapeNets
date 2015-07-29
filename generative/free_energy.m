function e = free_energy(model, data, l)
% free energy of RBM. This function deals two occasions of final layer RBM
% and the others.
% google it in http://www.deeplearning.net

if l < length(model.layers) || model.classes == 0
    e =  data * model.layers{l}.b';
    vwh = bsxfun(@plus, data * model.layers{l}.w, model.layers{l}.c);
    huge = vwh > 80;
    vwh_temp = vwh;
    vwh_temp(huge) = 0;
    e = e + sum(log(1 + exp(vwh_temp)), 2);
    vwh_temp = vwh;
    vwh_temp(~huge) = 0;
    e = e + sum(vwh_temp,2);
else
    if ~isfield('model','duplicate')
        model.duplicate = 1;
    end
    e =  data * model.layers{l}.b';
    e = e + (model.duplicate-1) * data(:,1:model.classes) * model.layers{l}.b(1:model.classes)';
    temp_w = model.layers{l}.w;
    temp_w(1:model.classes,:) = temp_w(1:model.classes,:) * model.duplicate;
    vwh = bsxfun(@plus, data * temp_w, model.layers{l}.c);
    huge = vwh > 80;
    vwh_temp = vwh;
    vwh_temp(huge) = 0;
    e = e + sum(log(1 + exp(vwh_temp)), 2);
    vwh_temp = vwh;
    vwh_temp(~huge) = 0;
    e = e + sum(vwh_temp,2); 
end
