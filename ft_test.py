import data as d
import train
import utils
import torchvision


learning_rate = .01
momentum =.09
weight_decay = .01
multistep = 1
train_loader = d.get_train_loader()
backbone = 'InceptionV3'


ssd300 = SSD300(backbone=ResNet(backbone, ''))
args.learning_rate = args.learning_rate * args.N_gpu * (args.batch_size / 32)
start_epoch = 0
iteration = 0
loss_func = Loss(dboxes)



optimizer = torch.optim.SGD(ssd300, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay)
scheduler = MultiStepLR(optimizer=optimizer, milestones=multistep, gamma=0.1)


inv_map = {v: k for k, v in val_dataset.label_map.items()}

total_time = 0


mean, std = generate_mean_std(args)

for epoch in range(start_epoch, args.epochs):
    start_epoch_time = time.time()
    scheduler.step()
    iteration = train_loop_func(ssd300, loss_func, epoch, optimizer, train_loader, val_dataloader, encoder, iteration,
                                logger, args, mean, std)
    end_epoch_time = time.time() - start_epoch_time
    total_time += end_epoch_time


    if epoch in args.evaluation:
        acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)

        if args.local_rank == 0:
            logger.update_epoch(epoch, acc)

    train_loader.reset()

print('total training time: {}'.format(total_time))
