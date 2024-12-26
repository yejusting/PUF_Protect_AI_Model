% boxplot([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10], 'Labels', {'10', '20', '30', '40', '50', '60', '70', '80', '90', '100'})


% x=0:0.01:100
% y=56.82 *(0<=x)
% plot(x,y)
% 
% hold on

x = result * 100
boxplot(x, 'Labels', {'10', '20', '30', '40', '50', '60', '70', '80', '90', '100'})
ylabel('Prediction Accuracy (%)')
% ylabel('p_d(%)')
xlabel('\itP_o_b_f \rm(%)')
