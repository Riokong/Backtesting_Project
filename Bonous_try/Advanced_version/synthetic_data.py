import torch
import torch.nn as nn
import numpy as np
from Advanced_version.models import Generator, Discriminator


class FinancialGAN:
    def __init__(self, input_dim, hidden_dim, sequence_length ):
        self.generator = Generator(input_dim, hidden_dim, sequence_length * 3)
        self.discriminator = Discriminator(sequence_length * 3, hidden_dim)
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001)
        self.criterion = nn.BCELoss()
    
    def train(self, real_data, epochs=1000, batch_size=32):
        for epoch in range(epochs):
            # Train Discriminator
            self.d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.generator.net[0].in_features)
            fake_data = self.generator(z).detach()
            
            idx = np.random.randint(0, len(real_data), batch_size)
            real_batch = torch.FloatTensor(real_data[idx])
            
            real_labels = torch.ones(batch_size, 1)
            d_real_loss = self.criterion(self.discriminator(real_batch), real_labels)
            
            fake_labels = torch.zeros(batch_size, 1)
            d_fake_loss = self.criterion(self.discriminator(fake_data), fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.generator.net[0].in_features)
            fake_data = self.generator(z)
            g_loss = self.criterion(self.discriminator(fake_data), real_labels)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            if epoch % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')
    
    def generate_samples(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.generator.net[0].in_features)
            samples = self.generator(z).numpy()
        return samples

class SyntheticDataGenerator:
    def __init__(self, real_data, sequence_length=30):
        self.real_data = real_data
        self.sequence_length = sequence_length
        input_size = sequence_length * real_data.shape[1]
        self.gan = FinancialGAN(
            input_dim=100,
            hidden_dim=256,
            sequence_length=sequence_length )
        self.mean = None
        self.std = None
        
    def prepare_training_data(self):
        sequences = []
        for i in range(len(self.real_data) - self.sequence_length):
            seq = self.real_data[i:i + self.sequence_length].flatten()
            sequences.append(seq)
        return np.array(sequences)
        
    def train(self, epochs=5000, batch_size=256):
        training_sequences = self.prepare_training_data()
        self.mean = np.mean(training_sequences)
        self.std = np.std(training_sequences)
        normalized_data = (training_sequences - self.mean) / self.std
        self.gan.train(normalized_data, epochs, batch_size)


        
    def generate_synthetic_data(self, n_samples):

        samples = self.gan.generate_samples(n_samples)
        return samples * self.std + self.mean
       