h = 0.01;
a = pi-2;
b = pi+2;
X = a:h:b;
Y = a:h:b;
z = 2.34;
n = length(X);
F = zeros(n);
i=0;
j=0;
for y = a:h:b
    i = i+1;
    for x = a:h:b
        j = j+1;
%         x
%         y
        F(i,j)=drill(x,y);
    end
    j = 0;
end
mesh(X,Y,F)
hold on;
for i = 1:max_iter
    Z1(i) = drill(X1(i),Y1(i));
    Z2(i) = drill(X2(i),Y2(i));
    Z4(i) = drill(X4(i),Y4(i));
end
% plot3(X1,Y1,Z1, 'ko', 'MarkerEdgeColor','b') %amsgrad
% plot3(X2,Y2,Z2, 'ko', 'MarkerEdgeColor','r') %nosadam

plot3(X4,Y4,Z4, 'ko', 'MarkerEdgeColor','b');
% figure;
% plot(1:max_iter,LR5,1:max_iter, LR3,'LineStyle','--','LineWidth',1.3)
% legend('AMSGrad', 'NosAdam')
% ylabel('x axis noise scale')
% xlabel('iterations')
% 
% figure;
% plot(1:max_iter,LR6,1:max_iter, LR4,'LineStyle','--','LineWidth',1.3)
% legend('AMSGrad', 'NosAdam')
% ylabel('y axis noise scale')
% xlabel('iterations')


    
