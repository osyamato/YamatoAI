

import Foundation

struct Tokenizer {

    func tokenize(_ text: String) -> [String] {

        return Array(text).map(String.init)

    }

}
