
        # 定义测试函数
        def test2(test_loader, epoch):
            encoder_model.eval()  # 设置模型为评估模式
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            preds_list = []    # 存储分类预测标签
            data_list = []     # 存储真实标签
            embeds_list = []   # 存储用于聚类的嵌入向量

            # 遍历所有测试数据（按 batch）
            for batch_idx, data in enumerate(test_loader):
                # 将每个 batch 中的数据移到指定设备（如 GPU）
                data = [d.to(device) for d in data]
                g, _, _ = encoder_model(data)
                gi = encoder_model.encoder.instance_projector(g)
                # 假设 forward_cluster 返回 (分类 logits, 嵌入向量)
                output = encoder_model.forward_cluster(data)
                
                # 计算交叉熵损失（对整个 batch 求和）
                loss = F.cross_entropy(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                
                # 获取分类预测（取最大概率的索引）
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                data_list += data[4].tolist()
                preds_list += pred.tolist()
                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
                
                # 收集嵌入向量（用于聚类评价）
                embeds_list.append(gi.detach().cpu())
            
            # 将各个 batch 的结果合并
            labels = torch.Tensor(data_list)
            preds = torch.Tensor(preds_list)
            all_embeds = torch.cat(embeds_list, dim=0)

            time_iter = time.time() - start
            test_loss /= n_samples
            acc = 100. * correct / n_samples

            # ---------------------------
            # 分类指标计算（原有部分）
            # ---------------------------
            classnums = 21
            r = recall(preds, labels.view_as(preds), classnums)
            p = precision(preds, labels.view_as(preds), classnums)
            f1 = f1_score(preds, labels.view_as(preds), classnums)
            fp = false_positive(preds, labels.view_as(preds), classnums)
            fn = false_negative(preds, labels.view_as(preds), classnums)
            tp = true_positive(preds, labels.view_as(preds), classnums)
            tn = true_negative(preds, labels.view_as(preds), classnums)

            r = (r.numpy()).round(7)
            p = (p.numpy()).round(7)
            f1 = (f1.numpy()).round(7)
            fp = fp.numpy()
            fn = fn.numpy()
            tp = tp.numpy()
            tn = tn.numpy()
            print('test_test_recall', " ".join('%s' % id for id in r))
            print('test_test_precision', " ".join('%s' % id for id in p))
            print('test_test_F1', " ".join('%s' % id for id in f1))

            conf_matrix = get_confusion_matrix(labels.view_as(preds), preds)
            plt.figure(figsize=(26, 26), dpi=60)
            plot_confusion_matrix(conf_matrix, classnums, epoch)

            print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                epoch, test_loss, correct, n_samples, acc))
            
            # ---------------------------
            # 聚类评价部分
            # ---------------------------
            # 将嵌入向量转为 numpy 数组，便于 KMeans 聚类
            embeds_np = all_embeds.numpy()
            true_labels_np = np.array(data_list)
            
            # 聚类的类别数设置为真实标签数
            n_clusters = len(np.unique(true_labels_np))
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeds_np)
            cluster_assignments = kmeans.labels_
            
            # 调用 evaluate 函数计算聚类指标
            nmi, ari, f, cluster_acc = evaluate(true_labels_np, cluster_assignments)
            print("聚类评价指标:")
            print("NMI: {:.4f}".format(nmi))
            print("ARI: {:.4f}".format(ari))
            print("Fowlkes-Mallows: {:.4f}".format(f))
            print("聚类准确率: {:.4f}".format(cluster_acc))
            
            # 返回分类准确率和聚类指标
            return acc, (nmi, ari, f, cluster_acc)

                
        def test1(test_loader, epoch):
            encoder_model.eval()
            start = time.time()
            test_loss, correct, n_samples = 0, 0, 0
            preds_list = []
            data_list = []
            for batch_idx, data in enumerate(test_loader):
                data = [d.to(args.device) for d in data]
                #g, _, _ = encoder_model(data.x, data.edge_index)
                #output = model(data)
                output = encoder_model.forward_cluster(data)
                loss = F.cross_entropy(output, data[4], reduction='sum')
                test_loss += loss.item()
                n_samples += len(output)
                pred = output.detach().cpu().max(1, keepdim=True)[1]
                data_list += data[4].tolist()
                preds_list += pred.tolist()
                correct += pred.eq(data[4].detach().cpu().view_as(pred)).sum().item()
            labels = torch.Tensor(data_list)
            preds = torch.Tensor(preds_list)
    
            time_iter = time.time() - start
    
            test_loss /= n_samples
            acc = 100. * correct / n_samples
    
            # 使用 torch_geometric.utils 计算评价指标
            classnums = 21
            r = recall(preds, labels.view_as(preds), classnums)
            p = precision(preds, labels.view_as(preds), classnums)
            f1 = f1_score(preds, labels.view_as(preds), classnums)
            fp = false_positive(preds, labels.view_as(preds), classnums)
            fn = false_negative(preds, labels.view_as(preds), classnums)
            tp = true_positive(preds, labels.view_as(preds), classnums)
            tn = true_negative(preds, labels.view_as(preds), classnums)
    
            r = (r.numpy()).round(7)
            p = (p.numpy()).round(7)
            f1 = (f1.numpy()).round(7)
            fp = fp.numpy()
            fn = fn.numpy()
            tp = tp.numpy()
            tn = tn.numpy()
            print('test_test_recall', " ".join('%s' % id for id in r))
            print('test_test_precision', " ".join('%s' % id for id in p))
            print('test_test_F1', " ".join('%s' % id for id in f1))
    
            conf_matrix = get_confusion_matrix(labels.view_as(preds), preds)
            plt.figure(figsize=(26, 26), dpi=60)
            plot_confusion_matrix(conf_matrix, classnums, epoch)
    
            print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, 
                                                                                                  test_loss, 
                                                                                                  correct, 
                                                                                                  n_samples, acc))
            return acc