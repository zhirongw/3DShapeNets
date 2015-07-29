function show_sample(samples)
% a simple visualization tool using isosurface to show each 3D sample.

n = size(samples,1);
for i = 1 : n
    the_sample = squeeze(samples(i,:,:,:,:));
    
    figure;
    p = patch(isosurface(the_sample,0.05));
    set(p,'FaceColor','red','EdgeColor','none');
    daspect([1,1,1])
    view(3); axis tight
    camlight 
    lighting gouraud;
    axis off;
    set(gcf,'Color','white');
    set(gca,'position',[0,0,1,1],'units','normalized');
    axis tight;
    %title(i);
    pause;
    close(gcf);
end
