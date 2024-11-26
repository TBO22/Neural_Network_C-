#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstdlib> 

using namespace std;

template <typename T>
class LinkedList {
public:
    struct Node {
        T value;
        Node* next;
        Node(T val) : value(val), next(nullptr) {}
    };

    Node* head;

    LinkedList() : head(nullptr) {}

    void append(T value) {
        Node* newNode = new Node(value);
        if (!head) {
            head = newNode;
        } else {
            Node* temp = head;
            while (temp->next) {
                temp = temp->next;
            }
            temp->next = newNode;
        }
    }

    int size() const {
        int count = 0;
        Node* temp = head;
        while (temp) {
            count++;
            temp = temp->next;
        }
        return count;
    }

    T get(int index) const {
        Node* temp = head;
        int count = 0;
        while (temp) {
            if (count == index) return temp->value;
            temp = temp->next;
            count++;
        }
    }

    void update(int index, T value) {
        Node* temp = head;
        int count = 0;
        while (temp) {
            if (count == index) {
                temp->value = value;
                return;
            }
            temp = temp->next;
            count++;
        }
    }

    LinkedList<T> copy() const {
        LinkedList<T> newList;
        Node* temp = head;
        while (temp) {
            newList.append(temp->value);
            temp = temp->next;
        }
        return newList;
    }
};

// Activation function enum
enum ActivationType {
    Sigmoid,
    ReLU
};

// Layer class with flexible neuron count and activation function
class Layer {
private:
    LinkedList<double> weights;  // Store weights for each neuron
    LinkedList<double> neurons;  // Store outputs of neurons
    double bias;                 // Bias for the layer
    ActivationType activationType;

    // Activation functions
    double sigmoid(double x) const {
        return 1.0 / (1.0 + exp(-x));
    }

    double relu(double x) const {
        return max(0.0, x);
    }

public:
    Layer(int inputSize, int neuronCount, ActivationType actType = Sigmoid)
        : activationType(actType), bias(((double)rand() / RAND_MAX) - 0.5) {
        // Initialize weights for each neuron
        for (int i = 0; i < neuronCount * inputSize; i++) {
            weights.append(((double)rand() / RAND_MAX) - 0.5);
        }

        // Initialize neurons (outputs)
        for (int i = 0; i < neuronCount; i++) {
            neurons.append(0.0);
        }
    }

    // Forward pass through the layer
    LinkedList<double> forward(const LinkedList<double>& inputs) {
        LinkedList<double> outputs;
        int inputSize = inputs.size();
        int neuronCount = neurons.size();

        // For each neuron in the layer
        for (int n = 0; n < neuronCount; n++) {
            double weightedSum = 0.0;

            // Compute weighted sum for this neuron
            for (int i = 0; i < inputSize; i++) {
                weightedSum += inputs.get(i) * weights.get(n * inputSize + i);
            }

            // Add bias
            weightedSum += bias;

            // Apply activation function
            double output;
            if (activationType == Sigmoid) {
                output = sigmoid(weightedSum);
            } else { // ReLU
                output = relu(weightedSum);
            }

            // Store neuron output
            neurons.update(n, output);
            outputs.append(output);
        }

        return outputs;
    }

    // Getters and setters
    LinkedList<double>& getWeights() { 
        return weights; 
        }
    double getBias() const { 
        return bias; 
        }
    void updateWeight(int index, double newWeight) { 
        weights.update(index, newWeight); 
        }
    void updateBias(double newBias) { 
        bias = newBias; 
        }
    ActivationType getActivationType() const { 
        return activationType; 
        }
};

// Neural Network class to manage layers
class NeuralNetwork {
private:
    LinkedList<Layer*> layers;  // Linked list of layers

public:
NeuralNetwork(int inputSize, LinkedList<Layer*>& layerList) {
    auto* temp = layerList.head; // Use the head of the LinkedList<Layer*>

    // Add layers
    while (temp) {
        layers.append(temp->value); // Append the layer to the internal LinkedList
        temp = temp->next;         // Move to the next node
    }
}

    // Forward propagation through all layers
    double predict(LinkedList<double>& inputs) {
        LinkedList<double> currentInputs = inputs.copy();
        auto* temp = layers.head;
        while (temp) {
            currentInputs = temp->value->forward(currentInputs);
            temp = temp->next;
        }
        return currentInputs.get(0);
    }

    // Training method (simple gradient descent)
void train(LinkedList<double>& inputs, double target, double learningRate) {
    double output = predict(inputs);
    double error = target - output;

    // Backpropagate through layers
    auto temp = layers.head;
    while (temp) {
        Layer* currentLayer = reinterpret_cast<Layer*>(temp->value);
        LinkedList<double>& currentWeights = currentLayer->getWeights();

        // Update weights for each neuron in the layer
        for (int i = 0; i < currentWeights.size(); i++) {
            double currentWeight = currentWeights.get(i);
            // Compute gradient of weights and apply to update the weights
            double gradient = error * inputs.get(i); // Simplified gradient (depends on activation and layer)
            double newWeight = currentWeight + learningRate * gradient;
            currentLayer->updateWeight(i, newWeight);
        }

        // Update bias (should also consider the error signal here)
        double newBias = currentLayer->getBias() + learningRate * error;  // Update bias using the error
        currentLayer->updateBias(newBias);

        temp = temp->next;
    }
}
// Print network parameters
void printParameters() {
    auto temp = layers.head;
    int layerIndex = 0;
    while (temp) {
        Layer* currentLayer = reinterpret_cast<Layer*>(temp->value);

        cout << "Layer " << layerIndex << " Weights: ";
        auto weightTemp = currentLayer->getWeights().head;
        while (weightTemp) {
            cout << weightTemp->value << " ";
            weightTemp = weightTemp->next;
        }
        cout << endl;

        cout << "Layer " << layerIndex << " Bias: " << currentLayer->getBias()
             << ", Activation: " << (currentLayer->getActivationType() == Sigmoid ? "Sigmoid" : "ReLU")
             << endl;

        temp = temp->next;
        layerIndex++;
    }
}
};
int main() {
    // Create layers with specified number of neurons and activation functions
    Layer firstLayer(2, 2, ReLU);       // 2 inputs, 4 neurons, ReLU

    LinkedList<Layer*> layersList; // Creating layerslist object to store all the layers
    layersList.append(&firstLayer);

    // Create neural network with 2 inputs, and the layers defined above
    NeuralNetwork nn(2, layersList);

    // Training data for AND logic
    double inputs[4][2] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}};
    double targets[4] = {2, 4, 6, 8};
    double learningRate = 0.1;
    int epochs = 10000000;

    // Train the network
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < 4; i++) {
            LinkedList<double> inputList;
            inputList.append(inputs[i][0]);
            inputList.append(inputs[i][1]);
            nn.train(inputList, targets[i], learningRate);
        }
    }

    // Test the network
    cout << "\nTesting Neural Network after " << epochs << " epochs:" << endl;
    for (int i = 0; i < 4; i++) {
        LinkedList<double> inputList;
        inputList.append(inputs[i][0]);
        inputList.append(inputs[i][1]);
        double output = nn.predict(inputList);
        cout << "Input: (" << inputs[i][0] << ", " << inputs[i][1]
             << ") -> Output: " << output << endl;
    }

    // Print final network parameters (weights and biases of all layers)
    nn.printParameters();
    cout<<"Testing on unseen data : "<<endl;
    double input_test[2][2] = {{5, 5},{6,6}};
  for (int i = 0; i < 2; i++) {
        LinkedList<double> inputList2;
        inputList2.append(input_test[i][0]);
        inputList2.append(input_test[i][1]);
        double output2 = nn.predict(inputList2);
        cout << "Input: (" << input_test[i][0] << ", " << input_test[i][1]
             << ") -> Output: " << output2 << endl;
    }
    return 0;
}
