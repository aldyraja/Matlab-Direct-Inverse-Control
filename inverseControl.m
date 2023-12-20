%inverseControl: prediksi x, hanya berdasarkan y, y-1, dan y-2. akan terbuat x-1 dan x-2
%                     x = f(y, y-1, y-2)
clear, clc;

%% //////////// Mempersiapkan Data ////////////
% ===== data sinyal sin
rentang = linspace(0, 15001, 15002)';
data_sin = zeros(15000, 6);
data_sin(:, 1) = rentang(3:15002, :);  % x
data_sin(:, 2) = rentang(2:15001, :);  % x-1
data_sin(:, 3) = sind(rentang(2:15001, :));  % y-1
data_sin(:, 4) = rentang(1:15000, :);  % x-2
data_sin(:, 5) = sind(rentang(1:15000, :));  % y-2
data_sin(:, 6) = sind(rentang(3:15002, :));  % y

% ===== data sinyal step
sys = tf(4,[1 2 10]);
[amplitude, time] = step(sys,1381.6);
data_step = zeros(15000, 6);
data_step(:, 1) = time(3:15002, :);  % x
data_step(:, 2) = time(2:15001, :);  % x-1
data_step(:, 3) = amplitude(2:15001, :);  % y-1
data_step(:, 4) = time(1:15000, :);  % x-2
data_step(:, 5) = amplitude(1:15000, :);  % y-2
data_step(:, 6) = amplitude(3:15002, :);  % y

% ===== data random
data_random = table2array(readtable('dataRandom_Normalized.csv','PreserveVariableNames',true));

% ===== data yang dipakai
data = data_random;
dataName = "data_random";

% ===== split data equally to train/validate/test
split_size = size(data, 1) * 1/3;

x_train = data(1:split_size , 2:6);  % | x-1 | y-1 | x-2 | y-2 | y |
y_train = data(1:split_size , 1);  % | x |

x_val = data(split_size+1:2*split_size, 2:6);
y_val = data(split_size+1:2*split_size, 1);

x_test = data(2*split_size+1:3*split_size, 2:6);
y_test = data(2*split_size+1:3*split_size, 1);

preparedDataFileName = "prep_invCtrl_" + dataName + ".mat";
save(preparedDataFileName, "x_train", "y_train", "x_val", "y_val", "x_test", "y_test")
% /////////////////////////////////////////////



%% ///////////////// TAHAP 1 //////////////////
% (output = x; input = y, y-1, x-1, y-2, x-2)

% dataName = "data_random";  % "data_random" / "data_sin" / "data_step"
preparedDataFileName = "prep_invCtrl_" + dataName + ".mat";
load(preparedDataFileName, "x_train", "y_train", "x_val", "y_val")

train_data_length = size(x_train, 1);
val_data_length = size(x_val, 1);

% =========== STEP 0: Inisialisasi ============
input = size(x_train, 2);

hiddenA = 6;
hiddenB = 4;

output = size(y_train, 2);


alpha = 0.2;      % NOTES: Berfungsi untuk mempercepat penurunan error, tetapi berpotensi menyebabkan osilasi. Punya Adid 0.1
Error_min = 10^(-7);
epoch_max = 10000;  %        Berfungsi untuk memperkecil error jika masih memungkinkan.
% =============================================


% == Metode Inisialisasi Bobot Nguyen-Widrow ==
% ===== Menentukan faktor skala
beta_uij = 0.7*power(hiddenA, 1/input);
beta_vjk = 0.7*power(hiddenB, 1/hiddenA);
beta_wkl = 0.7*power(output, 1/hiddenB);

% ===== Inisialisasi bobot secara random
min_val = -0.5;
max_val = 0.5;
uij = min_val + (max_val - min_val) .* rand(input, hiddenA);  % Generate an input-by-hiddenA column vector of uniformly distributed random numbers in the interval (min_val,max_val).
vjk = min_val + (max_val - min_val) .* rand(hiddenA, hiddenB);
wkl = min_val + (max_val - min_val) .* rand(hiddenB, output);

% ===== Hitung nilai norma euclidean untuk vektor Uj, Vk, dan Wl (||Uj||, ||Vk||, dan ||Wl||)
norm_uj = zeros(1, hiddenA);
norm_vk = zeros(1, hiddenB);
norm_wl = zeros(1, output);
for j = 1:hiddenA
     sumOfSquares = 0;
     for i = 1:input
     sumOfSquares = sumOfSquares + uij(i, j)^2;
     end
     norm_uj(j) = sqrt(sumOfSquares);
end
for k = 1:hiddenB
     sumOfSquares = 0;
     for j = 1:hiddenA
     sumOfSquares = sumOfSquares + vjk(j, k)^2;
     end
     norm_vk(k) = sqrt(sumOfSquares);
end
for l = 1:output
     sumOfSquares = 0;
     for k = 1:hiddenB
     sumOfSquares = sumOfSquares + wkl(k, l)^2;
     end
     norm_wl(l) = sqrt(sumOfSquares);
end

% ===== Update bobot uij, vjk, wkl
for j = 1:hiddenA
     for i = input
     uij(i, j) = beta_uij*uij(i, j)/norm_uj(j);
     end
end
for k = 1:hiddenB
     for j = hiddenA
     vjk(j, k) = beta_vjk*vjk(j, k)/norm_vk(k);
     end
end
for l = 1:output
     for k = 1:hiddenB
     wkl(k, l) = beta_wkl*wkl(k, l)/norm_wl(l);
     end
end

% ===== set bias uoj, vok, dan wol
uoj = -beta_uij + (beta_uij - (-beta_uij)) .* rand(1, hiddenA);  % Generate a 1-by-hiddenA column vector of uniformly distributed random numbers in the interval (-beta_uij,beta_uij).
vok = -beta_vjk + (beta_vjk - (-beta_vjk)) .* rand(1, hiddenB);
wol = -beta_wkl + (beta_wkl - (-beta_wkl)) .* rand(1, output);
% =============================================


% ======== for error plotting purposes ========
close all;
figure;
ax1 = subplot(2,1,1); ax1.YGrid = "on"; ax1.XGrid = "on";
ax2 = subplot(2,1,2); ax2.YGrid = "on"; ax2.XGrid = "on";
loss = [];
loss1 = [];
count = [];
% ===== Error plotting
Errors = zeros(1, epoch_max);
Errors2 = zeros(1, epoch_max);
% =============================================


% ========== Definisi fungsi tangenh ==========
syms tangenh(x)
tangenh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
tangenh = matlabFunction(tangenh);

% ===== turunan tangenh(x)
syms tangenh2(x)
tangenh2(x) = diff(tanh(x));
tangenh2 = matlabFunction(tangenh2);
% =============================================


% Time elapsed
tic;

% STEP 1:
for epoch = 1:epoch_max
    % Each data loop
    % STEP 2:
    xx = zeros(1, input);
    yy = zeros(1, output);
    sum_mse = 0;
    sum_mse2 = 0;
    for p = 1:train_data_length
 
        % Step 3:
        % Set the columns to train
        xx = double(x_train(p, :));  % menerima input xi
        yy = double(y_train(p, :));
        
        % Step 4:
        % ========== vektor ==========
        z_inj = zeros(1, hiddenA);
        zj = zeros(1, hiddenA);
        for o = 1:hiddenA
            z_inj(1, o) = uoj(1, o) + (xx * uij(:, o));
            zj(1, o) = double(tangenh(z_inj(1, o)));  % Activation function
        end
 
        % Step 5:
        % ========== vektor ==========
        z_ink = zeros(1, hiddenB);
        zk = zeros(1, hiddenB);
        for m = 1:hiddenB
            z_ink(1, m) = vok(1, m) + (zj * vjk(:, m));
            zk(1, m) = double(tangenh(z_ink(1, m)));  % Activation function
        end
        
        % Step 6:
        % ========== vektor ==========
        y_inl = zeros(1, output);
        yl = zeros(1, output);
        for l = 1:output
            y_inl(1, l) = wol(1, l) + (zk * wkl(:, l));
            yl(1, l) = double(tangenh(y_inl(1, l)));  % Activation function
        end
        
        % Error
        Error = (yy-yl).^2;
        sum_mse = sum_mse + Error;    % NOTES: beda metode dengan sebelum-sebelumnya,
        %                             %        sebelumnya pakai function sum() membungkus ((yy-yl).^2)
        
        % Backprop of error
        do_l = (yy - yl) .* double(tangenh2(y_inl));
        % Step 7:
        % ========== vektor ==========
        delta_wkl = zeros(hiddenB, output);
        delta_wol = zeros(1, output);
        for l = 1:output
            delta_wkl(:, l) = alpha * do_l(1, l) * zk';  %  weight output
            delta_wol(1, l) = alpha * do_l(1, l);  %  bias output
        end
        
        do_ink = do_l * wkl';  % menghitung semua koreksi error
        do_k = do_ink .* double(tangenh2(z_ink));  %  aktivasi koreksi error
        
        % Step 8:
        % ========== vektor ==========
        delta_vjk = zeros(hiddenA, hiddenB);
        delta_vok = zeros(1, hiddenB);
        for m = 1:hiddenB
            delta_vjk(:, m) = alpha * do_k(1, m) * zj';  %  weight hiddenB
            delta_vok(1, m) = alpha .* do_k(1, m);  %  bias hiddenB
        end
        
        do_inj = do_k * vjk';  % menghitung semua koreksi error
        do_j = do_inj .* double(tangenh2(z_inj));  %  aktivasi koreksi error
        
        % Step 9:
        % ========== vektor ==========
        delta_uij = zeros(input, hiddenA);
        delta_uoj = zeros(1, hiddenA);
        for o = 1:hiddenA
            delta_uij(:, o) = alpha * do_j(1, o) * xx';  %  bobot unit hidden
            delta_uoj(1, o) = alpha .* do_j(1, o);  %  error bias unit hidden
        end
        
        % Step 10:
        % ========== vektor ==========
        % Weight update
        for l = 1:output
            wkl(:, l) = wkl(:, l) + delta_wkl(:, l);  % weight output update
            wol(1, l) = wol(1, l) + delta_wol(1, l);  % bias output update
        end
        for m = 1:hiddenB
            vjk(:, m) = vjk(:, m) + delta_vjk(:, m);  % weight hidden update
            vok(1, m) = vok(1, m) + delta_vok(1, m);  % bias hidden update
        end
        for o = 1:hiddenA
            uij(:, o) = uij(:, o) + delta_uij(:, o);  % weight hidden update
            uoj(1, o) = uoj(1, o) + delta_uoj(1, o);  % bias hidden update
        end
    end

    for p = 1:val_data_length
        % Set the columns to train
        xx = double(x_val(p, :));
        yy = double(y_val(p, :));
        
        % Calculate hidden layer input
        % Activation function
        % z_inj = voj + (xx * vij);
        % zj = double(sigmoid(z_inj));
        % ========== diubah menjadi vektor ==========
        % Step 4 : Setiap unit / neuron hidden (Zj, j = 1,…,m)
        z_inj = zeros(1, hiddenA);
		zj = zeros(1, hiddenA);
        for o = 1:hiddenA
            z_inj(1, o) = uoj(1, o) + (xx * uij(:, o));
            zj(1, o) = double(tangenh(z_inj(1, o)));  % Activation function
        end
 
        % Step 5:
        % ========== vektor ==========
		z_ink = zeros(1, hiddenB);
		zk = zeros(1, hiddenB);
        for m = 1:hiddenB
            z_ink(1, m) = vok(1, m) + (zj * vjk(:, m));
            zk(1, m) = double(tangenh(z_ink(1, m)));  % Activation function
        end
        
        % Step 6:
        % ========== vektor ==========
        y_inl = zeros(1, output);
		yl = zeros(1, output);
        for l = 1:output
            y_inl(1, l) = wol(1, l) + (zk * wkl(:, l));
            yl(1, l) = double(tangenh(y_inl(1, l)));  % Activation function
        end

        Error2 = (yy-yl).^2;
        sum_mse2 = sum_mse2 + Error2;    % NOTES: beda metode dengan sebelum-sebelumnya,
        %                                %        sebelumnya pakai function sum() membungkus ((yy-yl).^2)
    end

    Errors = 0.5 * sum_mse;  % NOTES: sebelumnya Errors()  = Error
    Errors2 = 0.5 * sum_mse2;  %        sebelumnya Errors2() = Error2

    fprintf("Epoch : %d\t Training Loss: %d\t Validation Loss : %d\n", epoch, Errors, Errors2);

    if mod(epoch, 2) == 0
        loss = [loss, Errors];
        count = [count, epoch];
        loss1 = [loss1, Errors2];
        
        % Plot kedua kurva pada satu axes
        plot(ax1, count, loss, 'LineWidth', 2, 'DisplayName', 'Train Loss');
        hold(ax1, 'on'); % Menahan plot untuk menambahkan kurva selanjutnya
        plot(ax1, count, loss1, 'LineWidth', 2, 'DisplayName', 'Validation Loss');
        hold(ax1, 'off'); % Melepaskan penahanan plot
        
        ax1.YGrid = "on";
        ax1.XGrid = "on";
        title(ax1, 'Tahap 1');
        xlabel(ax1, 'Epoch');
        ylabel(ax1, 'Loss function');
        legend(ax1); % Menampilkan legenda
        
        drawnow();
    end
end

fprintf("Training ends at epoch = %d with Error = %d\n", epoch, Errors);
% /////////////////////////////////////////////



%% ///////////////// TAHAP 2 //////////////////
% (output = x, input = y, y-1, y-2)
% (melanjutkan bobot dari tahap 1)

% dataName = "data_random";  % "data_random" / "data_sin" / "data_step"
% preparedDataFileName = "prep_invCtrl_" + dataName + ".mat";
% load(preparedDataFileName, "x_train", "y_train", "x_val", "y_val")
% load("inverseControlOutputBest.mat")

train_data_length = size(x_train, 1);
val_data_length = size(x_val, 1);

% ===== Mempersiapkan input untuk tahap 2 =====
x_train2 = x_train;
x_train2(:, 1) = zeros(1, size(x_train2, 1));  %kosongkan x(t-1)
x_train2(:, 3) = zeros(1, size(x_train2, 1));  %kosongkan x(t-2)
% =============================================

data_x_pred = zeros(size(y_train, 1) , size(y_train, 2));      %NOTES: menyiapkan tempat untuk x hasil prediksi

% =========== STEP 0: Inisialisasi ============
% alpha = 0.2;      % NOTES: alpha boleh berbeda dengan NN tahap 1, karena training yang berlangsung berbeda
Error_min = Error_min/100;
% epoch_max = 250;  %        epoch_max boleh berbeda dengan NN tahap 1, karena training yang berlangsung berbeda
% =============================================


% ======== for error plotting purposes ========
loss2 = [];
loss3 = [];
count2 = [];
% Error plotting
Errors = zeros(1, epoch_max);
Errors2 = zeros(1, epoch_max);
% =============================================


% ========== Definisi fungsi tangenh ==========
syms tangenh(x)
tangenh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
tangenh = matlabFunction(tangenh);

% ===== turunan tangenh(x)
syms tangenh2(x)
tangenh2(x) = diff(tanh(x));
tangenh2 = matlabFunction(tangenh2);
% =============================================


% Time elapsed
tic;
 
% STEP 1:
for epoch = 1:epoch_max
    % Each data loop
    xx = zeros(1, input);
    yy = zeros(1, output);
    sum_mse = 0;
    sum_mse2 = 0;
    % STEP 2:
    for p = 1:train_data_length
 
        if p > 1
            %NOTES: Masukkan yl sebelumnya sebagai x(t-1) dan x(t-2)
            x_train2(p,1) = yl;
            if p > 2
                x_train2(p,3) = x_train2(p-1,1);
            end
        end
        
        % Step 3:
        % Set the columns to train
        xx = double(x_train2(p, :));  % menerima input xi
        yy = double(y_train(p, :));
        
        % Step 4:
        % ========== vektor ==========
        z_inj = zeros(1, hiddenA);
        zj = zeros(1, hiddenA);
        for o = 1:hiddenA
            z_inj(1, o) = uoj(1, o) + (xx * uij(:, o));
            zj(1, o) = double(tangenh(z_inj(1, o)));  % Activation function
        end
 
        % Step 5:
        z_ink = zeros(1, hiddenB);
        zk = zeros(1, hiddenB);
        for m = 1:hiddenB
            z_ink(1, m) = vok(1, m) + (zj * vjk(:, m));
            zk(1, m) = double(tangenh(z_ink(1, m)));  % Activation function
        end
        
        % Step 6:
        % ========== vektor ==========
        y_inl = zeros(1, output);
        yl = zeros(1, output);
        for l = 1:output
            y_inl(1, l) = wol(1, l) + (zk * wkl(:, l));
            yl(1, l) = double(tangenh(y_inl(1, l)));  % Activation function
        end
        
        % Error
        Error = (yy-yl).^2;
        sum_mse = sum_mse + Error;    % NOTES: beda metode dengan sebelum-sebelumnya,
        %                             %        sebelumnya pakai function sum() membungkus ((yy-yl).^2)
        
        % Backprop of error
        do_l = (yy - yl) .* double(tangenh2(y_inl));
        % Step 7:
        % ========== vektor ==========
        delta_wkl = zeros(hiddenB, output);
        delta_wol = zeros(1, output);
        for l = 1:output
            delta_wkl(:, l) = alpha * do_l(1, l) * zk';  %  weight output
            delta_wol(1, l) = alpha * do_l(1, l);  %  bias output
        end
        
        do_ink = do_l * wkl';  % menghitung semua koreksi error
        do_k = do_ink .* double(tangenh2(z_ink));  %  aktivasi koreksi error
        
        % Step 8:
        % ========== vektor ==========
        delta_vjk = zeros(hiddenA, hiddenB);
        delta_vok = zeros(1, hiddenB);
        for m = 1:hiddenB
            delta_vjk(:, m) = alpha * do_k(1, m) * zj';  %  weight hiddenB
            delta_vok(1, m) = alpha .* do_k(1, m);  %  bias hiddenB
        end
        
        do_inj = do_k * vjk';  % menghitung semua koreksi error
        do_j = do_inj .* double(tangenh2(z_inj));  %  aktivasi koreksi error
        
        % Step 9:
        delta_uij = zeros(input, hiddenA);
        delta_uoj = zeros(1, hiddenA);
        for o = 1:hiddenA
            delta_uij(:, o) = alpha * do_j(1, o) * xx';  %  bobot unit hidden
            delta_uoj(1, o) = alpha .* do_j(1, o);  %  error bias unit hidden
        end
        
        % Step 10 : Setiap unit output (Yk, k = 1,…,l)
        % ========== vektor ==========
        % Weight update
        for l = 1:output
            wkl(:, l) = wkl(:, l) + delta_wkl(:, l);  % weight output update
            wol(1, l) = wol(1, l) + delta_wol(1, l);  % bias output update
        end
        for m = 1:hiddenB
            vjk(:, m) = vjk(:, m) + delta_vjk(:, m);  % weight hidden update
            vok(1, m) = vok(1, m) + delta_vok(1, m);  % bias hidden update
        end
        for o = 1:hiddenA
            uij(:, o) = uij(:, o) + delta_uij(:, o);  % weight hidden update
            uoj(1, o) = uoj(1, o) + delta_uoj(1, o);  % bias hidden update
        end
    
        data_x_pred(p) = yl;
        
    end


    for p = 1:val_data_length
        % Set the columns to train
        xx = double(x_val(p, :));
        yy = double(y_val(p, :));
        
        % Calculate hidden layer input
        % Activation function
        % z_inj = voj + (xx * vij);
        % zj = double(sigmoid(z_inj));
        % ========== diubah menjadi vektor ==========
        % Step 4 : Setiap unit / neuron hidden (Zj, j = 1,…,m)
        z_inj = zeros(1, hiddenA);
		zj = zeros(1, hiddenA);
        for o = 1:hiddenA
            z_inj(1, o) = uoj(1, o) + (xx * uij(:, o));
            zj(1, o) = double(tangenh(z_inj(1, o)));  % Activation function
        end
 
        % Step 5:
		z_ink = zeros(1, hiddenB);
		zk = zeros(1, hiddenB);
        for m = 1:hiddenB
            z_ink(1, m) = vok(1, m) + (zj * vjk(:, m));
            zk(1, m) = double(tangenh(z_ink(1, m)));  % Activation function
        end
        
        % Step 6:
        % ========== vektor ==========
        y_inl = zeros(1, output);
		yl = zeros(1, output);
        for l = 1:output
            y_inl(1, l) = wol(1, l) + (zk * wkl(:, l));
            yl(1, l) = double(tangenh(y_inl(1, l)));  % Activation function
        end
        
        Error2 = (yy-yl).^2;
        sum_mse2 = sum_mse2 + Error2;    % NOTES: beda metode dengan sebelum-sebelumnya,
        %                                %        sebelumnya pakai function sum()membungkus ((yy-yl).^2)
    end

    Errors = 0.5 * sum_mse;    % NOTES: sebelumnya Errors() = Error
    Errors2 = 0.5 * sum_mse2;  %        sebelumnya Errors2() = Error2

    fprintf("Epoch : %d\t Training Loss: %d\t Validation Loss : %d\n",epoch, Errors,Errors2);

    if mod(epoch, 2) == 0
        loss2 = [loss2, Errors];
        count2 = [count2, epoch];
        loss3 = [loss3, Errors2];
        
        % Plot kedua kurva pada satu axes
        plot(ax2, count2, loss2, 'LineWidth', 2, 'DisplayName', 'Train Loss');
        hold(ax2, 'on'); % Menahan plot untuk menambahkan kurva selanjutnya
        plot(ax2, count2, loss3, 'LineWidth', 2, 'DisplayName', 'Validation Loss');
        hold(ax2, 'off'); % Melepaskan penahanan plot
        
        ax2.YGrid = "on";
        ax2.XGrid = "on";
        title(ax2, 'Tahap 2');
        xlabel(ax2, 'Epoch');
        ylabel(ax2, 'Loss function');
        legend(ax2); % Menampilkan legenda
        
        drawnow();
    end
end

fprintf("Training ends at epoch = %d with Error = %d\n", epoch, Errors);

save("inverseControlOutput.mat", "x_train2", "data_x_pred", "uoj", "uij", "vok", "vjk", "wol", "wkl")
% /////////////////////////////////////////////



%% /////////// Actual vs Predicted ////////////
figure(2);
plot_x = 1:100;

plot_y1 = data(1:100 , 1);
plot_y2 = data_x_pred(1:100);

plot(plot_x, plot_y1, 'b', 'DisplayName', 'Actual Data');
hold on;

plot(plot_x, plot_y2, 'r', 'DisplayName', 'Predicted Data');
hold off;

xlabel('x');
ylabel('y');

legend;
% /////////////////////////////////////////////



%% ///////////// Training Error Logging /////////////
fname='inverseControlLog.xlsx';
T = table(hiddenA,hiddenB,alpha,epoch,Errors);
writetable(T,fname,'WriteMode','Append','WriteVariableNames',false,'WriteRowNames',true)
% /////////////////////////////////////////////