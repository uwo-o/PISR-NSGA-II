#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <random>

// --- Representación de Árbol (Propuesto en el Paper) ---
class Node {
public:
    virtual ~Node() = default;
    // Evaluación recursiva post-orden
    virtual double evaluate(double x, double y) const = 0;
    // Diferenciación simbólica automatizada nativa (Algoritmo 1)
    virtual double derivative(double x, double y, char respect_to) const = 0;
    virtual void mutate_erc(std::mt19937& gen) = 0;
    virtual void print() const = 0;
};

// Nodos Terminales: Variables (x, y)
class VariableNode : public Node {
    char var_name;
public:
    VariableNode(char name) : var_name(name) {}
    
    double evaluate(double x, double y) const override {
        return (var_name == 'x') ? x : y;
    }
    
    double derivative(double x, double y, char respect_to) const override {
        return (var_name == respect_to) ? 1.0 : 0.0; // Caso base de derivada
    }
    
    void mutate_erc(std::mt19937& gen) override {} // No aplica
    void print() const override { std::cout << var_name; }
};

// Nodos Terminales: Constantes Aleatorias Efímeras (ERC)
class ERCNode : public Node {
    double value;
    std::normal_distribution<double> gaussian_noise;
public:
    ERCNode(double val) : value(val), gaussian_noise(0.0, 0.1) {}
    
    double evaluate(double x, double y) const override { return value; }
    
    double derivative(double x, double y, char respect_to) const override {
        return 0.0; // Derivada de una constante es 0
    }
    
    // Mutación Paramétrica: Perturbación Gaussiana
    void mutate_erc(std::mt19937& gen) override {
        value += gaussian_noise(gen); 
    }
    
    void print() const override { std::cout << value; }
};

// Nodos Internos: Operadores (+, *)
class AddNode : public Node {
    std::unique_ptr<Node> left, right;
public:
    AddNode(std::unique_ptr<Node> l, std::unique_ptr<Node> r) 
        : left(std::move(l)), right(std::move(r)) {}
        
    double evaluate(double x, double y) const override {
        return left->evaluate(x, y) + right->evaluate(x, y);
    }
    
    double derivative(double x, double y, char respect_to) const override {
        // Regla de la suma: d(u+v) = du + dv
        return left->derivative(x, y, respect_to) + right->derivative(x, y, respect_to);
    }
    
    void mutate_erc(std::mt19937& gen) override {
        left->mutate_erc(gen);
        right->mutate_erc(gen);
    }
    
    void print() const override {
        std::cout << "("; left->print(); std::cout << " + "; right->print(); std::cout << ")";
    }
};

// --- Bucle Principal de Ejemplo ---
int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    // Construyendo el cromosoma: u(x,y) = x + 4.5 (Ejemplo inspirado en Fig 1)
    std::unique_ptr<Node> var_x = std::make_unique<VariableNode>('x');
    std::unique_ptr<Node> erc_const = std::make_unique<ERCNode>(4.5);
    std::unique_ptr<Node> tree = std::make_unique<AddNode>(std::move(var_x), std::move(erc_const));

    std::cout << "--- Metodo Propuesto PIGP ---" << std::endl;
    std::cout << "Ecuacion original: ";
    tree->print();
    std::cout << "\nEvaluacion en (x=2.0, y=1.0): " << tree->evaluate(2.0, 1.0);
    std::cout << "\nDerivada exacta d/dx: " << tree->derivative(2.0, 1.0, 'x');

    // Mutación paramétrica (ERC)
    tree->mutate_erc(gen);
    std::cout << "\n\nDespues de Mutacion Parametrica (Gaussiana en ERC):" << std::endl;
    std::cout << "Ecuacion mutada: ";
    tree->print();
    std::cout << std::endl;

    return 0;
}