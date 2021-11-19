import run

right = 0
total = 0
for step, data in enumerate(train_loader):
    more_toxic_ids = data['more_toxic_ids'].to(device, dtype = torch.long)
    more_toxic_mask = data['more_toxic_mask'].to(device, dtype = torch.long)
    less_toxic_ids = data['less_toxic_ids'].to(device, dtype = torch.long)
    less_toxic_mask = data['less_toxic_mask'].to(device, dtype = torch.long)
    targets = data['target'].to(device, dtype=torch.long)
        
    batch_size = more_toxic_ids.size(0)
    print(batch_size)

    more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
    print(more_toxic_outputs)
    less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)
    print(less_toxic_outputs)
    right += (more_toxic_outputs>less_toxic_outputs).sum()
    print(right)
    total += batch_size 
        
    loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)
    print(loss)
    break