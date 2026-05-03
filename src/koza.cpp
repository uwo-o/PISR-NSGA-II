#include <iostream>
#include <vector>
#include <string>
#include <random>

// --- Método Estilo Koza usando Gramática BNF ---
// Gramática de ejemplo:
// <expr> ::= <expr> <op> <expr> | ( <expr> ) | <var> | <const>
// <op>   ::= + | - | *
// <var>  ::= x | y

class BNFMapper {
public:
    int max_depth = 5;
    
    // Mapea un arreglo de enteros (codones) a una expresión simbólica
    std::string map_genotype_to_phenotype(const std::vector<int>& codons, int& codon_index, int depth, std::string symbol) {
        if (codon_index >= codons.size()) return ""; // Genotipo inválido o corto

        int current_codon = codons[codon_index++];

        if (symbol == "<expr>") {
            if (depth >= max_depth) {
                // Forzar terminal para evitar recursión infinita
                return (current_codon % 2 == 0) ? map_genotype_to_phenotype(codons, codon_index, depth + 1, "<var>")
                                                : map_genotype_to_phenotype(codons, codon_index, depth + 1, "<const>");
            }
            
            int rule = current_codon % 3; // 3 reglas posibles para <expr>
            if (rule == 0) return map_genotype_to_phenotype(codons, codon_index, depth + 1, "<expr>") + 
                                  map_genotype_to_phenotype(codons, codon_index, depth + 1, "<op>") + 
                                  map_genotype_to_phenotype(codons, codon_index, depth + 1, "<expr>");
            else if (rule == 1) return map_genotype_to_phenotype(codons, codon_index, depth + 1, "<var>");
            else return map_genotype_to_phenotype(codons, codon_index, depth + 1, "<const>");
        }
        else if (symbol == "<op>") {
            int rule = current_codon % 3;
            if (rule == 0) return "+";
            if (rule == 1) return "-";
            return "*";
        }
        else if (symbol == "<var>") {
            return (current_codon % 2 == 0) ? "x" : "y";
        }
        else if (symbol == "<const>") {
            return std::to_string((current_codon % 10) + 1); // Constantes rígidas 1-10
        }
        return "";
    }
};

struct IndividualBNF {
    std::vector<int> genotype;
    std::string phenotype; // Ecuación matemática resultante
    double mse_error;
};

// --- Bucle Principal de Ejemplo ---
int main() {
    std::vector<int> sample_codons = {12, 5, 22, 9, 31, 2, 8, 14, 7, 10};
    int index = 0;
    
    BNFMapper mapper;
    std::string equation = mapper.map_genotype_to_phenotype(sample_codons, index, 0, "<expr>");
    
    std::cout << "--- Metodo BNF (Koza) ---" << std::endl;
    std::cout << "Genotipo (Codones): ";
    for(int c : sample_codons) std::cout << c << " ";
    std::cout << "\nFenotipo (Ecuacion Generada): " << equation << std::endl;
    
    return 0;
}