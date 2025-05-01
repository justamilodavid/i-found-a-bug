function RealDiabetesPredictor()
    % 1. Initialization
    clc; close all; 
    warning('off', 'stats:glmfit:IterationLimit');
    warning('off', 'stats:glmfit:BadScaling');
    
    % Load and preprocess data
    fprintf('Loading dataset...\n');
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv';
    try
        % Read the raw data as text
        rawData = webread(url);
        
        % Split into lines and then into numbers
        dataLines = splitlines(rawData);
        numLines = length(dataLines);
        data = zeros(numLines, 9); % 8 features + 1 outcome
        
        for i = 1:numLines
            if ~isempty(dataLines{i})
                data(i,:) = str2double(strsplit(dataLines{i}, ','));
            end
        end
        
        % Remove any empty lines and create table
        data = data(~isnan(data(:,1)),:);
        data = array2table(data, 'VariableNames', {'Pregnancies','Glucose','BP','SkinThickness',...
                               'Insulin','BMI','DPF','Age','Outcome'});
    catch ME
        error('Failed to load dataset: %s', ME.message);
    end
    
    X = data{:,1:8};
    Y = data.Outcome; % Keep as numeric for now
    
    % Handle missing values and outliers
    fprintf('Preprocessing data...\n');
    X(X==0) = NaN;
    for i = 1:size(X,2)
        X(isnan(X(:,i)),i) = median(X(~isnan(X(:,i)),i),'omitnan');
    end
    
    % Feature engineering
    X(:,9) = X(:,2).*X(:,6); % Glucose-BMI interaction
    X(:,10) = X(:,5)./X(:,6); % Insulin-to-BMI ratio
    featureNames = [data.Properties.VariableNames(1:8), {'Glucose_BMI', 'Insulin_BMI'}];
    
    % Cross-validation setup with stratification
    k = 10;
    cv = cvpartition(Y,'KFold',k,'Stratify',true);
    
    % Feature selection using MRMR
    fprintf('Performing feature selection...\n');
    [idx, scores] = fscmrmr(X, Y);
    selected_features = idx(1:min(8, length(idx))); % Keep top 8 features
    X = X(:,selected_features);
    featureNames = featureNames(selected_features);
    
    % All models with improved configuration
    models = {
        'Decision Tree', @(X,Y) fitctree(X,Y,'MaxNumSplits',20,'MinParentSize',10);
        'Logistic Regression', @(X,Y) fitglm(X,Y,'Distribution','binomial','Link','logit');
        'SVM', @(X,Y) fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true,'BoxConstraint',1);
        % In the models cell array, modify these lines:
    % ... other models ...
    'Random Forest', @(X,Y) TreeBagger(100,X,Y,'Method','classification','OOBPrediction','on','OOBPredictorImportance','on','MinLeafSize',5);
    'Bagged Trees', @(X,Y) TreeBagger(50,X,Y,'Method','classification','OOBPrediction','on','OOBPredictorImportance','on','MinLeafSize',5);
    % ... other models ...
  'Naive Bayes', @(X,Y) fitcnb(X,Y,'DistributionNames','kernel','Width',0.1);
        'XGBoost', @(X,Y) trainXGBoost(X,Y);
    };
    
    % Initialize metrics structure
    fprintf('Initializing metrics...\n');
    numModels = size(models, 1);
    metrics = struct('Name', cell(numModels, 1), ...
                   'Accuracy', cell(numModels, 1), ...
                   'Accuracy_Std', cell(numModels, 1), ...
                   'Precision', cell(numModels, 1), ...
                   'Recall', cell(numModels, 1), ...
                   'F1', cell(numModels, 1), ...
                   'AUC', cell(numModels, 1), ...
                   'Time', cell(numModels, 1), ...
                   'CM', cell(numModels, 1), ...
                   'Model', cell(numModels, 1)); % Store trained models
    
    % Cross-validation loop with proper data handling
    fprintf('Starting cross-validation...\n');
    for fold = 1:k
        fprintf('\nProcessing fold %d/%d...\n', fold, k);
        trainIdx = training(cv,fold);
        testIdx = test(cv,fold);
        
        % Apply ADASYN only to training data
        [X_train_res, Y_train_res] = resampleDataADASYN(X(trainIdx,:), Y(trainIdx));
        
        % Normalize using training data stats only
        [X_train_norm, mu, sigma] = zscore(X_train_res);
        X_test_norm = (X(testIdx,:) - mu)./sigma;
        Y_train = Y_train_res; % Keep as numeric
        Y_test = Y(testIdx); % Keep as numeric
        
        for m = 1:numModels
            fprintf('  Training %s...\n', models{m,1});
            try
                tic;
                currentModel = models{m,2}(X_train_norm, Y_train);
                trainTime = toc;
                
                % Store trained model
                if fold == 1
                    metrics(m).Model = {currentModel};
                else
                    metrics(m).Model = [metrics(m).Model; {currentModel}];
                end
                
                % Model-specific prediction handling
                switch models{m,1}
                    case 'Logistic Regression'
                        scores = predict(currentModel, X_test_norm);
                        Y_pred = double(scores > 0.5);
                        scores = scores(:);
                    case {'Decision Tree', 'Naive Bayes'}
                        [Y_pred, scores] = predict(currentModel, X_test_norm);
                        Y_pred = double(Y_pred);
                        scores = scores(:,2);
                    case 'SVM'
                        [~, scores] = predict(currentModel, X_test_norm);
                        Y_pred = double(scores(:,2) > 0);
                        scores = scores(:,2);
                    case {'Random Forest', 'Bagged Trees'}
                        [Y_pred_cell, scores] = predict(currentModel, X_test_norm);
                        Y_pred = str2double(Y_pred_cell);
                        scores = scores(:,2);
                    case 'XGBoost'
                        [Y_pred, scores] = predictXGBoost(currentModel, X_test_norm);
                        Y_pred = double(Y_pred > 0.5);
                        scores = scores(:);
                end
                
                % Calculate confusion matrix
                cm = confusionmat(Y_test, Y_pred);
                
                % Calculate metrics
                tp = cm(2,2);
                fp = cm(1,2);
                fn = cm(2,1);
                tn = cm(1,1);
                
                accuracy = (tp + tn) / sum(cm(:));
                precision = tp / (tp + fp + eps);
                recall = tp / (tp + fn + eps);
                f1 = 2 * (precision * recall) / (precision + recall + eps);
                [~,~,~,auc] = perfcurve(Y_test, scores, 1);
                
                % Store metrics
                if fold == 1
                    metrics(m).Name = models{m,1};
                    metrics(m).Accuracy = accuracy;
                    metrics(m).Precision = precision;
                    metrics(m).Recall = recall;
                    metrics(m).F1 = f1;
                    metrics(m).AUC = auc;
                    metrics(m).Time = trainTime;
                    metrics(m).CM = {cm};
                else
                    metrics(m).Accuracy = [metrics(m).Accuracy; accuracy];
                    metrics(m).Precision = [metrics(m).Precision; precision];
                    metrics(m).Recall = [metrics(m).Recall; recall];
                    metrics(m).F1 = [metrics(m).F1; f1];
                    metrics(m).AUC = [metrics(m).AUC; auc];
                    metrics(m).Time = [metrics(m).Time; trainTime];
                    metrics(m).CM = [metrics(m).CM; {cm}];
                end
                
            catch ME
                fprintf('Error in model %s: %s\n', models{m,1}, ME.message);
                % Assign NaN values to indicate failure
                if fold == 1
                    metrics(m).Name = models{m,1};
                    metrics(m).Accuracy = NaN;
                    metrics(m).Precision = NaN;
                    metrics(m).Recall = NaN;
                    metrics(m).F1 = NaN;
                    metrics(m).AUC = NaN;
                    metrics(m).Time = NaN;
                    metrics(m).CM = {NaN(2,2)};
                    metrics(m).Model = {NaN};
                else
                    metrics(m).Accuracy = [metrics(m).Accuracy; NaN];
                    metrics(m).Precision = [metrics(m).Precision; NaN];
                    metrics(m).Recall = [metrics(m).Recall; NaN];
                    metrics(m).F1 = [metrics(m).F1; NaN];
                    metrics(m).AUC = [metrics(m).AUC; NaN];
                    metrics(m).Time = [metrics(m).Time; NaN];
                    metrics(m).CM = [metrics(m).CM; {NaN(2,2)}];
                    metrics(m).Model = [metrics(m).Model; {NaN}];
                end
            end
        end
    end
    
    % Calculate mean and std for metrics
    for m = 1:numModels
        metrics(m).Accuracy_Mean = mean(metrics(m).Accuracy, 'omitnan');
        metrics(m).Accuracy_Std = std(metrics(m).Accuracy, 'omitnan');
        metrics(m).Precision_Mean = mean(metrics(m).Precision, 'omitnan');
        metrics(m).Recall_Mean = mean(metrics(m).Recall, 'omitnan');
        metrics(m).F1_Mean = mean(metrics(m).F1, 'omitnan');
        metrics(m).AUC_Mean = mean(metrics(m).AUC, 'omitnan');
        metrics(m).Time_Mean = mean(metrics(m).Time, 'omitnan');
        
        % Calculate average confusion matrix
        validFolds = ~cellfun(@(x) any(isnan(x(:))), metrics(m).CM);
        if any(validFolds)
            avgCM = mean(cat(3, metrics(m).CM{validFolds}), 3);
            metrics(m).AvgCM = round(avgCM);
        else
            metrics(m).AvgCM = NaN(2,2);
        end
    end
    
    % Display results
    fprintf('\n=== Cross-Validated Performance (Mean ± Std) ===\n');
    resultTable = table(...
        {metrics.Name}', ...
        [metrics.Accuracy_Mean]', ...
        [metrics.Accuracy_Std]', ...
        [metrics.Precision_Mean]', ...
        [metrics.Recall_Mean]', ...
        [metrics.F1_Mean]', ...
        [metrics.AUC_Mean]', ...
        [metrics.Time_Mean]', ...
        'VariableNames', {'Model', 'Accuracy', 'Accuracy_Std', 'Precision', 'Recall', 'F1', 'AUC', 'Time'});
    
    % Remove models that failed completely
    validModels = ~isnan([metrics.AUC_Mean]);
    resultTable = resultTable(validModels,:);
    validMetrics = metrics(validModels);
    
    % Sort by AUC and display
    disp(sortrows(resultTable,'AUC','descend'));
    
    % Plot confusion matrices
    figure('Name','Average Confusion Matrices','Position',[100 100 1200 800]);
    numValidModels = sum(validModels);
    rows = ceil(numValidModels/3);
    cols = min(3, numValidModels);
    
    for m = 1:numValidModels
        subplot(rows, cols, m);
        if ~any(isnan(validMetrics(m).AvgCM(:)))
            confusionchart(validMetrics(m).AvgCM, {'Negative', 'Positive'});
            title(sprintf('%s (Acc=%.2f)', validMetrics(m).Name, validMetrics(m).Accuracy_Mean));
        else
            text(0.5, 0.5, 'Model Failed', 'HorizontalAlignment', 'center');
            title(sprintf('%s (Failed)', validMetrics(m).Name));
        end
    end
    
    % Train final model on full resampled data
    if any(validModels)
        [X_resampled, Y_resampled] = resampleDataADASYN(X, Y);
        [X_norm, mu, sigma] = zscore(X_resampled);
        Y_numeric = Y_resampled;
        
        % Select best model
        [~, bestIdx] = max([metrics(validModels).AUC_Mean]);
        bestModelName = metrics(validModels(bestIdx)).Name;
        bestModelFunc = models{strcmp(models(:,1), bestModelName),2};
        
        fprintf('\nTraining final %s model on full dataset...\n', bestModelName);
        finalModel = bestModelFunc(X_norm, Y_numeric);
    else
        error('No valid models were successfully trained.');
    end
    
    % Generate Partial Dependence Plots for ALL models
    generatePartialDependencePlots(metrics(validModels), X_norm, featureNames);
    
    % Generate Feature Importance for ALL models
    generateFeatureImportance(metrics(validModels), X_norm, Y_numeric, featureNames);
    % Plot decision boundaries for all models
    plotDecisionBoundary(metrics(validModels), X_norm, Y_numeric, featureNames);
    % After cross-validation, before displaying results, add ROC curve plotting
    plotROCCurves(metrics(validModels), X_norm, Y_numeric);
    
        % Launch GUI with best model
    launchEnhancedGUI(finalModel, X_norm, X_resampled, featureNames, bestModelName);
    
      % After plotROCCurves() call
figure;
for m = 1:length(metrics)
    model = metrics(m).Model{1};
    [scores, Y_test] = getModelScores(model, metrics(m).Name, X_norm, Y_numeric);
    [prec,rec,~,auc] = perfcurve(Y_test,scores,1,'XCrit','reca','YCrit','prec');
    plot(rec,prec,'LineWidth',2,'DisplayName',sprintf('%s (AUPRC=%.2f)',metrics(m).Name,auc));
end
title('Precision-Recall Curves (Diabetes Prediction)');
xlabel('Recall (Sensitivity)'); ylabel('Precision (PPV)');
legend('Location','southwest'); grid on;
set(gca,'FontSize',12);
    
    % After all analyses
[cluster_idx,~] = kmeans(X_norm(:,1:min(5,end)),3); % Use top 5 features for clustering
perf_by_cluster = zeros(3, length(metrics));

figure;
for c = 1:3
    cluster_data = X_norm(cluster_idx==c,:);
    cluster_labels = Y_numeric(cluster_idx==c);
    for m = 1:length(metrics)
        [~,scores] = getModelScores(metrics(m).Model{1}, metrics(m).Name, cluster_data, cluster_labels);
        [~,~,~,auc] = perfcurve(cluster_labels, scores, 1);
        perf_by_cluster(c,m) = auc;
    end
end
heatmap({metrics.Name}, {'Cluster1','Cluster2','Cluster3'}, perf_by_cluster,...
        'Colormap',parula,'ColorLimits',[0.5 1]);
title('Model Performance by Patient Cluster');
ylabel('Patient Subgroups'); xlabel('Models');
end

function generatePartialDependencePlots(metrics, X_norm, featureNames)
    % Set up figure with professional styling
    fig = figure('Name','Partial Dependence Analysis','Position',[100 100 1400 900]);
    set(fig, 'Color', [1 1 1], 'InvertHardcopy', 'off');
    
    numModels = length(metrics);
    numFeatures = size(X_norm, 2);
    
    % Create tiled layout with proper spacing
    t = tiledlayout(numModels, numFeatures, 'TileSpacing', 'tight', 'Padding', 'compact');
    t.Title.String = '';
    t.Title.FontSize = 14;
    t.Title.FontWeight = 'bold';
    
    % Custom color palette
    colors = lines(numModels);
    
    for m = 1:numModels
        modelName = metrics(m).Name;
        model = metrics(m).Model{1};
        
        for f = 1:numFeatures
            nexttile;
            hold on;
            
            % Generate grid and reference data
            gridVals = linspace(min(X_norm(:,f)), max(X_norm(:,f)), 50)';
            X_pdp = repmat(median(X_norm), 50, 1);
            X_pdp(:,f) = gridVals;
            
            % Get predictions
            switch modelName
                case 'Logistic Regression'
                    preds = predict(model, X_pdp);
                case {'Decision Tree', 'Naive Bayes'}
                    [~, scores] = predict(model, X_pdp);
                    preds = scores(:,2);
                case 'SVM'
                    [~, scores] = predict(model, X_pdp);
                    preds = 1./(1+exp(-scores(:,2)));
                case {'Random Forest', 'Bagged Trees'}
                    [~, scores] = predict(model, X_pdp);
                    preds = scores(:,2);
                case 'XGBoost'
                    [~, scores] = predictXGBoost(model, X_pdp);
                    preds = scores;
            end
            
            % Create smooth plot with confidence bands
            [x_smooth, y_smooth, y_lower, y_upper] = smoothPDP(gridVals, preds);
            
            % Plot confidence band first
            fill([x_smooth; flipud(x_smooth)], [y_lower; flipud(y_upper)], ...
                 [0.7 0.7 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
            
            % Main PDP line
            plot(x_smooth, y_smooth, 'LineWidth', 2.5, 'Color', colors(m,:));
            
            % Formatting
            box on;
            grid on;
            set(gca, 'FontSize', 9, 'FontName', 'Arial', 'LineWidth', 1.5);
            ylim([0 1]);
            
            % Only show x labels on bottom row
            if m == numModels
                xlabel(strrep(featureNames{f}, '_', ' '), 'FontSize', 10, 'FontWeight', 'bold');
            end
            
            % Only show y labels on first column
            if f == 1
                ylabel({'Diabetes Probability'; modelName}, 'FontSize', 10, 'FontWeight', 'bold');
            end
            
            % Only show titles on top row
            if m == 1
                title(strrep(featureNames{f}, '_', ' '), 'FontSize', 11, 'FontWeight', 'bold');
            end
        end
    end
    
    % Add unified legend
    lg = legend(arrayfun(@(x) x.Name, metrics, 'UniformOutput', false));
    lg.Layout.Tile = 'north';
    lg.Orientation = 'horizontal';
    lg.FontSize = 11;
    lg.Box = 'off';
    
    % Export-ready settings
    set(fig, 'PaperPositionMode', 'auto', 'Color', 'w');
end

function generateFeatureImportance(metrics, X_norm, Y, featureNames)
    % Create high-quality importance plot
    fig = figure('Name','Feature Importance Analysis','Position',[100 100 1400 700]);
    set(fig, 'Color', [1 1 1], 'InvertHardcopy', 'off');
    
    numModels = length(metrics);
    numFeatures = size(X_norm, 2);
    
    % Create tiled layout
    t = tiledlayout(1, numModels, 'TileSpacing', 'tight', 'Padding', 'compact');
    t.Title.String = '';
    t.Title.FontSize = 14;
    t.Title.FontWeight = 'bold';
    
    % Custom color map
    cmap = parula(numFeatures);
    
    for m = 1:numModels
        nexttile;
        hold on;
        
        modelName = metrics(m).Name;
        model = metrics(m).Model{1};
        
        try
            % Calculate importance
            switch modelName
                case 'Logistic Regression'
                    imp = abs(model.Coefficients.Estimate(2:end));
                case 'Decision Tree'
                    imp = predictorImportance(model);
                case {'Random Forest', 'Bagged Trees'}
                    if isprop(model, 'OOBPermutedPredictorDeltaError')
                        imp = model.OOBPermutedPredictorDeltaError;
                    else
                        imp = predictorImportance(model);
                    end
                case 'XGBoost'
                    imp = predictorImportance(model);
                case 'SVM'
                    rng(42);
                    baseAcc = mean(predict(model, X_norm) == Y);
                    imp = zeros(1, numFeatures);
                    for f = 1:numFeatures
                        X_perm = X_norm;
                        X_perm(:,f) = X_perm(randperm(size(X_perm,1)), f);
                        permAcc = mean(predict(model, X_perm) == Y);
                        imp(f) = baseAcc - permAcc;
                    end
                case 'Naive Bayes'
                    if isa(model.DistributionParameters{1,1}, 'char')
                        imp = zeros(1, numFeatures);
                        for f = 1:numFeatures
                            mu1 = model.DistributionParameters{2,f}(1);
                            mu0 = model.DistributionParameters{1,f}(1);
                            sigma1 = model.DistributionParameters{2,f}(2);
                            sigma0 = model.DistributionParameters{1,f}(2);
                            imp(f) = abs((mu1 - mu0)/sqrt((sigma1^2 + sigma0^2)/2));
                        end
                    else
                        imp = ones(1, numFeatures);
                    end
            end
            
            % Normalize and sort
            imp = imp / max(imp);
            [sortedImp, idx] = sort(imp, 'descend');
            sortedNames = strrep(featureNames(idx), '_', ' ');
            
            % Create bar plot with custom styling
            h = barh(sortedImp, 'FaceColor', 'flat');
            for f = 1:numFeatures
                h.CData(f,:) = cmap(idx(f),:);
            end
            
            % Add value labels
            for f = 1:numFeatures
                text(sortedImp(f)+0.02, f, sprintf('%.2f', sortedImp(f)), ...
                    'FontSize', 9, 'FontWeight', 'bold');
            end
            
            % Formatting
            set(gca, 'YTick', 1:numFeatures, 'YTickLabel', sortedNames, ...
                'FontSize', 10, 'FontName', 'Arial', 'LineWidth', 1.5);
            xlim([0 1.1]);
            grid on;
            box on;
            title(modelName, 'FontSize', 12, 'FontWeight', 'bold');
            xlabel('Normalized Importance', 'FontSize', 10);
            
        catch ME
            text(0.5, 0.5, 'Importance Not Available', ...
                'HorizontalAlignment', 'center', 'FontSize', 12);
            title(modelName, 'FontSize', 12);
        end
    end
    
    % Add colorbar
    colormap(cmap);
    cb = colorbar;
    cb.Layout.Tile = 'east';
    cb.Label.String = 'Feature Rank';
    cb.FontSize = 10;
    cb.Ticks = linspace(0,1,numFeatures);
    cb.TickLabels = compose('%d',1:numFeatures);
    
    % Export settings
    set(fig, 'PaperPositionMode', 'auto', 'Color', 'w');
end
function plotDecisionBoundary(metrics, X, Y, featureNames)
    % Select top 2 features based on importance
    [~, scores] = fscmrmr(X, Y);
    [~, idx] = sort(scores, 'descend');
    topFeatures = idx(1:2);
    
    % Create figure
    figure('Name', 'Decision Boundaries', 'Position', [100 100 1200 800]);
    numModels = length(metrics);
    rows = ceil(numModels/3);
    cols = min(3, numModels);
    
    % Prepare data for plotting
    X_plot = X(:, topFeatures);
    feature1 = X_plot(:,1);
    feature2 = X_plot(:,2);
    
    % Create grid for decision surface
    x1 = linspace(min(feature1), max(feature1), 100);
    x2 = linspace(min(feature2), max(feature2), 100);
    [X1, X2] = meshgrid(x1, x2);
    X_grid = [X1(:), X2(:)];
    
    for m = 1:numModels
        model = metrics(m).Model{1}; % Get the first fold's model
        modelName = metrics(m).Name;
        
        subplot(rows, cols, m);
        hold on;
        
        % Prepare full feature matrix for prediction
        X_full = zeros(size(X_grid,1), size(X,2));
        X_full(:,topFeatures(1)) = X_grid(:,1);
        X_full(:,topFeatures(2)) = X_grid(:,2);
        
        % Fill other features with median values
        for f = 1:size(X,2)
            if ~ismember(f, topFeatures)
                X_full(:,f) = median(X(:,f));
            end
        end
        
        % Make predictions
        switch modelName
            case 'Logistic Regression'
                scores = predict(model, X_full);
                Y_grid = reshape(scores, size(X1));
            case {'Decision Tree', 'Naive Bayes'}
                [~, scores] = predict(model, X_full);
                Y_grid = reshape(scores(:,2), size(X1));
            case 'SVM'
                [~, scores] = predict(model, X_full);
                Y_grid = reshape(1./(1+exp(-scores(:,2))), size(X1));
            case {'Random Forest', 'Bagged Trees'}
                [~, scores] = predict(model, X_full);
                Y_grid = reshape(scores(:,2), size(X1));
            case 'XGBoost'
                [~, scores] = predictXGBoost(model, X_full);
                Y_grid = reshape(scores, size(X1));
        end
        
        % Plot decision boundary and probability contours
        contourf(X1, X2, Y_grid, 'LineWidth', 2);
        colormap(jet);
        colorbar;
        
        % Plot data points
        gscatter(feature1, feature2, Y, 'rb', 'o+', [], 'off');
        
        % Formatting
        title(modelName, 'FontSize', 12);
        xlabel(featureNames{topFeatures(1)}, 'FontSize', 10);
        ylabel(featureNames{topFeatures(2)}, 'FontSize', 10);
        legend('Negative', 'Positive', 'Location', 'best');
        grid on;
        axis tight;
    end
end 
function plotROCCurves(metrics, X, Y)
    % Create figure for ROC curves
    figure('Name', 'ROC Curve Comparison (Cross-Validated)', 'Position', [100 100 900 800]);
    hold on;
    
    % Colors for different models
    colors = lines(length(metrics));
    
    % Initialize legend entries
    legendEntries = cell(length(metrics), 1);
    
    % Store all fold curves for shading
    allXroc = cell(length(metrics), 1);
    allYroc = cell(length(metrics), 1);
    aucValues = cell(length(metrics), 1);
    
    % =============================================
    % Part 1: Plot individual fold curves (light)
    % =============================================
    for m = 1:length(metrics)
        modelName = metrics(m).Name;
        numFolds = length(metrics(m).Model);
        foldXroc = cell(numFolds, 1);
        foldYroc = cell(numFolds, 1);
        foldAuc = zeros(numFolds, 1);
        
        for fold = 1:numFolds
            model = metrics(m).Model{fold};
            
            % Get predictions for this fold's model
            [scores, Y_test] = getModelScores(model, modelName, X, Y);
            
            % Compute ROC curve
            [Xroc, Yroc, ~, auc] = perfcurve(Y_test, scores, 1);
            foldXroc{fold} = Xroc;
            foldYroc{fold} = Yroc;
            foldAuc(fold) = auc;
            
            % Plot individual fold curves (transparent)
            plot(Xroc, Yroc, 'Color', [colors(m,:) 0.2], 'LineWidth', 0.5);
        end
        
        allXroc{m} = foldXroc;
        allYroc{m} = foldYroc;
        aucValues{m} = foldAuc;
    end
    
    % =============================================
    % Part 2: Plot mean ± std curves (bold)
    % =============================================
    for m = 1:length(metrics)
        % Interpolate all folds to common FPR grid
        commonFPR = linspace(0, 1, 100);
        interpTPR = zeros(length(allYroc{m}), length(commonFPR));
        
        for fold = 1:length(allYroc{m})
            [~, uniqueIdx] = unique(allXroc{m}{fold});
            interpTPR(fold,:) = interp1(...
                allXroc{m}{fold}(uniqueIdx), ...
                allYroc{m}{fold}(uniqueIdx), ...
                commonFPR, 'linear', 0);
        end
        
        % Compute mean and std
        meanTPR = mean(interpTPR, 1);
        stdTPR = std(interpTPR, 0, 1);
        
        % Plot mean curve
        h(m) = plot(commonFPR, meanTPR, 'Color', colors(m,:), 'LineWidth', 3);
        
        % Plot std envelope
        fill([commonFPR, fliplr(commonFPR)], ...
             [meanTPR + stdTPR, fliplr(meanTPR - stdTPR)], ...
             colors(m,:), 'FaceAlpha', 0.2, 'EdgeColor', 'none');
         
        % Store legend entry
        meanAUC = mean(aucValues{m});
        legendEntries{m} = sprintf('%s (AUC = %.3f ± %.3f)', ...
            metrics(m).Name, meanAUC, std(aucValues{m}));
    end
    
    % =============================================
    % Formatting
    % =============================================
    % Plot random classifier
    plot([0 1], [0 1], 'k--', 'LineWidth', 1.5);
    
    % Labels and titles
    xlabel('False Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('True Positive Rate', 'FontSize', 12, 'FontWeight', 'bold');
    title('Cross-Validated ROC Curves (Mean ± 1 STD)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Legend and grid
    legend(h, legendEntries, 'Location', 'southeast', 'FontSize', 10);
    grid on;
    axis square;
    xlim([0 1]);
    ylim([0 1]);
    set(gca, 'FontSize', 11, 'LineWidth', 1.5);
    
    % Add AUC table annotation
    aucMatrix = cell2mat(aucValues');
    [~, bestIdx] = max(mean(aucMatrix));
    annotation('textbox', [0.15 0.15 0.3 0.1], ...
        'String', sprintf('Best Model: %s\nAUC = %.3f ± %.3f', ...
        metrics(bestIdx).Name, mean(aucMatrix(bestIdx,:)), std(aucMatrix(bestIdx,:))), ...
        'FitBoxToText', 'on', 'BackgroundColor', 'white', ...
        'EdgeColor', colors(bestIdx,:), 'LineWidth', 2);
  
    
end

% Helper function to get model scores consistently
function [scores, Y_test] = getModelScores(model, modelName, X, Y)
    switch modelName
        case 'Logistic Regression'
            scores = predict(model, X);
            Y_test = Y;
        case {'Decision Tree', 'Naive Bayes'}
            [~, scores] = predict(model, X);
            scores = scores(:,2);
            Y_test = Y;
        case 'SVM'
            [~, scores] = predict(model, X);
            scores = scores(:,2);
            Y_test = Y;
        case {'Random Forest', 'Bagged Trees'}
            [~, scores] = predict(model, X);
            scores = scores(:,2);
            Y_test = Y;
        case 'XGBoost'
            [~, scores] = predictXGBoost(model, X);
            Y_test = Y;
    end
end



function [x_smooth, y_smooth, y_lower, y_upper] = smoothPDP(x, y)
    % Create smoothed PDP with confidence bands
    [x_smooth, idx] = sort(x);
    y_smooth = smoothdata(y(idx), 'gaussian', 5);
    
    % Simple confidence band estimation
    window = 3;
    y_std = movstd(y(idx), window);
    y_lower = y_smooth - y_std;
    y_upper = y_smooth + y_std;
    
    % Ensure bounds are within [0,1]
    y_lower = max(y_lower, 0);
    y_upper = min(y_upper, 1);
end

function [X_resampled, Y_resampled] = resampleDataADASYN(X,Y)
    minorityClass = (Y==1);
    X_min = X(minorityClass,:);
    X_maj = X(~minorityClass,:);
    
    if isempty(X_min)
        X_resampled = X;
        Y_resampled = Y;
        return;
    end
    
    ratio = size(X_maj,1)/size(X_min,1);
    needed = max(0, round(ratio*size(X_min,1)) - size(X_min,1));
    synth = zeros(needed, size(X,2));
    
    if needed > 0
        D = pdist2(X_min, X_min);
        D(logical(eye(size(D)))) = inf;
        density = 1./min(D,[],2);
        density = density/sum(density);
        
        for j = 1:needed
            idx = randsample(size(X_min,1),1,true,density);
            [~,neighborIdx] = min(D(idx,:));
            synth(j,:) = X_min(idx,:) + rand(1,size(X,2)).*(X_min(neighborIdx,:)-X_min(idx,:));
        end
    end
    
    X_resampled = [X; synth];
    Y_resampled = [Y; ones(needed,1)];
end

function model = trainXGBoost(X, Y)
    model = fitcensemble(X, Y, 'Method', 'Bag', ...
                        'Learners', templateTree('MaxNumSplits', 20), ...
                        'NumLearningCycles', 100);
end

function [Y_pred, scores] = predictXGBoost(model, X)
    [Y_pred, scores] = predict(model, X);
    Y_pred = Y_pred > 0.5;
    scores = scores(:,2);
end

function launchEnhancedGUI(model, X_norm, X_original, featureNames, modelType)
    % Create uifigure instead of regular figure
    fig = uifigure('Position',[100 100 900 650],'Name','Diabetes Risk Assessment');
    
    % Add menu for saving figures
    m = uimenu(fig,'Text','File');
    m4 = uimenu(m,'Text','Exit','Separator','on','MenuSelectedFcn',@(src,event) close(fig));
    
    tg = uitabgroup(fig,'Position',[20 20 860 610]);
    predTab = uitab(tg,'Title','Risk Prediction');
    
    % Use uigridlayout with the uifigure
    glInput = uigridlayout(predTab,[6 4]);
    glInput.RowHeight = repmat({'fit'},1,6);
    glInput.ColumnWidth = [150 200 200 200];
    
    ranges = [min(X_original); max(X_original)]';
    edits = gobjects(length(featureNames),1);
    
    for i = 1:length(featureNames)
        row = ceil(i/3);
        col = mod(i-1,3)+2;
        
        uilabel(glInput,'Text',sprintf('%s (%d-%d):',featureNames{i},round(ranges(i,1)),round(ranges(i,2))),...
               'HorizontalAlignment','right');
        
        edits(i) = uieditfield(glInput,'numeric',...
                  'Limits',ranges(i,:),...
                  'Value',median(X_original(:,i)),...
                  'RoundFractionalValues','on');
    end
    
    btnPredict = uibutton(glInput,'Text','Predict Diabetes Risk',...
                 'ButtonPushedFcn',@(src,event) updatePrediction(),...
                 'BackgroundColor',[0.1 0.5 0.8],'FontColor','white');
    btnPredict.Layout.Row = 6;
    btnPredict.Layout.Column = [1 4];
    
    % Create a panel for results
    pnlResults = uipanel(predTab,'Position',[20 100 820 230]);
    
    % Create UI axes for the plot
    axRisk = uiaxes(pnlResults,'Position',[80 60 660 140]);
    title(axRisk,'Diabetes Risk Assessment');
    ylabel(axRisk,'Probability (%)');
    axRisk.YLim = [0 100];
    axRisk.XTick = [];
    
    lblResult = uilabel(pnlResults,'Position',[10 10 800 40],...
                'Text','','FontSize',18,'FontWeight','bold',...
                'HorizontalAlignment','center');
    
    infoTab = uitab(tg,'Title','Model Information');
    uitextarea(infoTab,'Position',[20 20 820 560],...
              'Value',sprintf('Model Type: %s\n\nTraining Date: %s\n\nFeatures Used:\n%s',...
              modelType,datestr(now),strjoin(featureNames,'\n')));
    
    function updatePrediction()
        input = zeros(1,length(featureNames));
        for i = 1:length(featureNames)
            input(i) = edits(i).Value;
        end
        
        input_norm = (input - min(X_original))./(max(X_original)-min(X_original));
        
        % Model-specific prediction handling
        switch modelType
            case 'Logistic Regression'
                scores = predict(model, input_norm);
                pred = scores > 0.5;
                risk = scores*100;
            case {'Decision Tree', 'Naive Bayes'}
                [~, scores] = predict(model, input_norm);
                pred = scores(:,2) > 0.5;
                risk = scores(:,2)*100;
            case 'SVM'
                [~, scores] = predict(model, input_norm);
                pred = scores(:,2) > 0;
                risk = (1./(1+exp(-scores(:,2))))*100;
            case {'Random Forest', 'Bagged Trees'}
                [~, scores] = predict(model, input_norm);
                pred = scores(:,2) > 0.5;
                risk = scores(:,2)*100;
            case 'XGBoost'
                [~, scores] = predictXGBoost(model, input_norm);
                pred = scores > 0.5;
                risk = scores*100;
            otherwise
                error('Unknown model type');
        end
        
        cla(axRisk);
        bar(axRisk,[risk, 100-risk],'stacked','FaceColor','flat');
        colormap(axRisk,[0.8 0 0; 0 0.6 0]);
        legend(axRisk,{'Diabetes Risk','Healthy'},'Location','northoutside');
        
        if pred
            resultText = sprintf('HIGH RISK (%.1f%) - Consult physician',risk);
            lblResult.FontColor = [0.8 0 0];
        else
            resultText = sprintf('Low Risk (%.1f%) - No action needed',risk);
            lblResult.FontColor = [0 0.6 0];
        end
        
        if risk > 70
            confidence = ' [High Confidence]';
        elseif risk > 30
            confidence = ' [Moderate Confidence]';
        else
            confidence = ' [Low Confidence]';
        end
        
        lblResult.Text = [resultText confidence];
    end
end