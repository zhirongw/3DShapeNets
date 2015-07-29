function [cols] = cubicle2col(vol,scale)
% convert a 3D volume to columns, i.e, each column is a small cube in the
% 3D volume. (An extension of im2col).
% vol: 3D volume.
% scale: the size of the small cube. The size of the volume should be
% divisible by scale.

if ndims(vol) < 3
    error('input data should be a volume');
end

a_vol = vol;
square_size = size(a_vol);
block_size = square_size(1:3) ./ scale;
assert(all(block_size == floor(block_size)));

cols = zeros([scale^2, numel(vol)./scale^3], class(vol));
for c = 1 : scale ^ 3
    x_off = mod(c-1 , scale); y_off = mod(floor((c-1) / scale), scale); z_off = floor((c-1) / scale^2 );
    per_channel = vol((x_off + 1) : scale : end, (y_off + 1) : scale : end , (z_off + 1) : scale : end, :);
    cols(c, :) = per_channel(:);
end
