import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import trange
from sklearn.metrics import classification_report


class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=4096):
        self.encodings = tokenizer(data['input'], truncation=True, padding=True,
                                 max_length=max_length, return_tensors='pt')
        self.labels = torch.tensor(data['output'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    
dataset = load_dataset("csv", data_files={"train":"datasets/fbr_train.csv","test":"datasets/fbr_test.csv","validation":"datasets/fbr_val.csv"})

# Initialize tokenizer and model
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

# Create datasets
train_dataset = TextDataset(dataset['train'], tokenizer)
val_dataset = TextDataset(dataset['validation'], tokenizer)
test_dataset = TextDataset(dataset['test'], tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=2)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# Training loop
num_epochs = 5

for epoch in trange(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}

        
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Average validation loss: {avg_val_loss:.4f}")
    print(f"Validation accuracy: {accuracy:.4f}")

# Final test set evaluation
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        test_loss += outputs.loss.item()

        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

test_accuracy = correct / total
print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")

model.save_pretrained('severity_mb')
tokenizer.save_pretrained('severity_mb')

test_loss = 0
correct = 0
total = 0
pred = []
gt = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        test_loss += outputs.loss.item()
        gt.extend(batch['labels'])
        predictions = torch.argmax(outputs.logits, dim=1)
        pred.extend(predictions)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

predt = []
for i in pred:
    predt.append(int(i))

gtt = []
for i in gt:
    gtt.append(int(i))

print(classification_report(gtt,predt))