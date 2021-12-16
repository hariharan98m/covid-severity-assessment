import torch
import torch.nn as nn
from model import CNN3d
from data import CTDataset, mosmed_dataloaders
from ignite.engine import Engine, Events
from ignite.metrics import MeanAbsoluteError
import pdb

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def train_step(engine, batch):
    model.train()
    optimizer.zero_grad()
    scan, y = batch[0].to(device), batch[2].to(device)
    scan = scan.permute(0, -1, 1, 2).unsqueeze(1).type(torch.float32)
    y_pred = model(scan)
    loss = criterion(y_pred, y)
    print(y_pred, y)
    loss.backward()
    optimizer.step()
    return {
        'loss': loss.item()
    }

def output_transform(output):
    # `output` variable is returned by above `process_function`
    y_pred = output['y_pred']
    y = output['y_true']
    return y_pred, y

trainer = Engine(train_step)

@trainer.on(Events.ITERATION_COMPLETED(every=1))
def iter_log_train(engine):
    print(f"{engine.state.epoch}/{engine.state.max_epochs} iter {engine.state.iteration} - loss : {engine.state.output['loss']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED(every=1))
def epoch_log_train(engine):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(f"training results - epoch : {engine.state.epoch} avg mse : {metrics['mae']:.2f}")

def validation_step(engine, batch):
    model.eval()
    with torch.no_grad():
        scan, y = batch[0].to(device), batch[2].to(device)
        y_pred = model(scan)
        return {
            'y': y,
            'y_pred': y_pred
        }

evaluator = Engine(validation_step)

@evaluator.on(Events.EPOCH_COMPLETED(every=1))
def epoch_log_val_results(engine):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(f"evaluation results - epoch : {engine.state.epoch} avg mse : {metrics['mae']:.2f}")

handler = ModelCheckpoint('/tmp/models', 'myprefix', n_saved=2, create_dir=True)

if __name__ == '__main__':
    model = CNN3d()
    model = model.to(device)
    train_loader, val_loader = mosmed_dataloaders(8, 1, 0.2)
    train_shapes = [scan.shape for scan, _, _, _ in train_loader]
    val_shapes = [scan.shape for scan, _, _, _ in val_loader]
    print(train_shapes)
    print(val_shapes)
    pdb.set_trace()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='mean')

    # create metrics
    mae = MeanAbsoluteError(output_transform = output_transform)
    mae.attach(evaluator, 'mae')
    trainer.run(train_loader, max_epochs = 2)