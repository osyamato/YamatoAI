import Foundation

struct Vocabulary {
    
    private(set) var wordToID: [String: Int] = [:]
    
    private(set) var idToWord: [String] = []
    
    var count: Int {

        idToWord.count

    }
    
    mutating func add(_ word: String) {
        
        if wordToID[word] != nil {
            
            return
            
        }
        
        let newID = idToWord.count
        
        wordToID[word] = newID
        
        idToWord.append(word)
        
    }
    
    mutating func build(from dataset: Dataset) {
        
        let tokenizer = Tokenizer()
        
        for example in dataset.examples {
            
            let inputTokens = tokenizer.tokenize(example.input)
            
            for token in inputTokens {
                
                add(token)
                
            }
            
            let outputTokens = tokenizer.tokenize(example.output)
            
            for token in outputTokens {
                
                add(token)
            }
            
        }
        
    }
}
