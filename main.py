import pickle
import os
import random
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
# from beepy import beep

def convert_to_windows(data, model):
    windows = []; w_size = model.n_window
    for i, g in enumerate(data): 
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)

def load_dataset(dataset, subdataset):
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = subdataset + '_' + file
        if dataset == 'SMAP': file = subdataset + '_' + file
        if dataset == 'MSL': file = subdataset + '_' + file
        if dataset == 'UCR': file = subdataset + '_' + file
        if dataset == 'NAB': file = subdataset + '_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    # loader = [i[:, debug:debug+1] for i in loader]
    if args.less: loader[0] = cut_array(0.2, loader[0])
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(modelname, dims):
    import src.models
    model_class = getattr(src.models, modelname)
    if modelname in ['SkipGramNS_Keras']:
        model = model_class(dims)
    else:
        model = model_class(dims).double()
    optimizer = None
    scheduler = None
    if model.name not in ['SkipGramNS', 'SkipGramNS_Keras']:
        # SkipGramNS model requires to define optimizer and sheduler after initialize embeddings 
        optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1; accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    l = nn.MSELoss(reduction = 'mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction = 'none')
        compute = ComputeLoss(model, 0.1, 0.005, 'cpu', model.n_gmm)
        n = epoch + 1; w_size = model.n_window
        l1s = []; l2s = []
        if training:
            for d in data:
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s = []
            for d in data: 
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    if 'Attention' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []; res = []
        if training:
            for d in data:
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data: 
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item()); klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.detach().numpy(), y_pred.detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item()); l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data: 
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1); ae2s.append(ae2); ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction = 'none')
        n = epoch + 1; w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            xs = []
            for d in data: 
                if 'MTAD_GAT' in model.name: 
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction = 'none')
        bcel = nn.BCELoss(reduction = 'mean')
        msel = nn.MSELoss(reduction = 'mean')
        real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
        real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        n = epoch + 1; w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d) 
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item()); gls.append(gl.item()); dls.append(dl.item())
                # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            outputs = []
            for d in data: 
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.detach().numpy(), y_pred.detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction = 'none')
        data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size = bs)
        n = epoch + 1; w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple): z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple): z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    elif model.name in ['SkipGramNS']:
        l = nn.MSELoss()
        n = epoch + 1
        if training:
            # SkipGramNS training process
            # Data contains the generated training skipgrmans
            for i, pair in enumerate(data[0][0]):
                # Prepare input
                target_word = torch.tensor(pair[0])
                context_word = torch.tensor(pair[1])
                y = torch.tensor(data[0][1][i])
                y_pred = model(target_word, context_word)
                loss = l(y_pred, y.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {loss.detach().numpy()}')
            return np.mean(loss.detach().numpy()), optimizer.param_groups[0]['lr']
        else:
            # SkipGramNS testing phase 
            y_pred = np.empty(data[0].shape[0])
            for i in range(0, data[0].shape[0]):
                target_word = torch.tensor(data[0][i])
                context_word = torch.tensor(data[1][i])
                y_pred[i] = model(target_word, context_word)
            return None, y_pred
    elif model.name in ['SkipGramNS_Keras']:
        if training:
            loss = model.train(data, epoch)
            return loss, None
        else:
            y_pred = model.evaluate(data)
            return None, y_pred.squeeze()
    else:
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.detach().numlabelspy(), y_pred.detach().numpy()

if __name__ == '__main__':
    ## Fix required random seeds
    random.seed(1)
    torch.manual_seed(1)
    tf.random.set_seed(1)
    np.random.seed(1)

    ## Load data and model
    train_loader, test_loader, labels = load_dataset(args.dataset, args.subdataset)
    if args.model in ['MERLIN']:
        eval(f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
    model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name: 
        trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    if model.name in ['SkipGramNS']:
        # Discretize train/test data and obtain vocabulary size
        skip_grams, trainD, testD, vocab_size = prepare_discretized_data(trainD, testD, model, debug=True)
        # Load SkipGramNS embbedings with proper size and initialize optimizer and scheduler
        model.init_emb(vocab_size + 1)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=model.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    if model.name in ['SkipGramNS_Keras']:
        #testD, labels = downsampling(testD, labels, factor=args.downsampling_rate)
        # Discretize train/test data and obtain vocabulary size
        skip_grams, trainD, testD, vocab_size = prepare_discretized_data_keras(trainD, testD, model, debug=True)
        model.init_model(vocab_size + 1)

    ### Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        if model.name in ['SkipGramNS_Keras']:
            loss, _ = backprop(5, model, skip_grams, trainO, optimizer, scheduler)
        else:
            num_epochs = 5; e = epoch + 1; start = time()
            for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
                if model.name in ['SkipGramNS']:
                    lossT, lr = backprop(e, model, skip_grams, trainO, optimizer, scheduler)
                else:
                    lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
                accuracy_list.append((lossT, lr))
            print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
            save_model(model, optimizer, scheduler, e, accuracy_list)
            plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    ### Testing phase
    if model.name not in ['SkipGramNS_Keras']:
        torch.zero_grad = True
        model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)

    ### Plot curves
    if not args.test:
        if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
        if model.name not in ['SkipGramNS', 'SkipGramNS_Keras']:
            plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

    ### Scores
    df = pd.DataFrame()
    lossT, yt_pred = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    if model.name in ['SkipGramNS', 'SkipGramNS_Keras']:
        labelsFinal = labelsFinal[1:]
        # With SkipGramNS we apply the POT over the perplexity scores
        yt_perplex = estimate_perplexity(yt_pred)
        y_perplex = estimate_perplexity(y_pred)
        result, pred = pot_eval(yt_perplex, y_perplex, labelsFinal)
        print('ytperplex -> mean:{}º   min:{}  max:{}'.format(yt_perplex.mean(), yt_perplex.min(), yt_perplex.max()))
        print('yperplex -> mean:{}º   min:{}  max:{}'.format(y_perplex.mean(), y_perplex.min(), y_perplex.max()))
    else:
        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
            result, pred = pot_eval(lt, l, ls); preds.append(pred)
            df = df.append(result, ignore_index=True)
        # preds = np.concatenate([i.reshape(-1, 1) + 0 for i in preds], axis=1)
        # pd.DataFrame(preds, columns=[str(i) for i in range(10)]).to_csv('labels.csv')
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
        result.update(hit_att(loss, labels))
        result.update(ndcg(loss, labels))
        print(df)
    pprint(result)
    # pprint(getresults2(df, result))
    # beep(4)
