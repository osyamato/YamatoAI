import Foundation

struct Softmax {

    func calculate(
        scores: [Float]
    ) -> [Float] {

        
        guard !scores.isEmpty else {

                    return []

                }
        // ① 総和を求める
        var total: Float = 0

        for index in scores.indices {

            total += exp(scores[index])

        }

        // ② 確率を入れる配列
        var probabilities: [Float] = []

        // ③ 確率を計算して追加
        for index in scores.indices {

            let probability =

                exp(scores[index]) / total

            probabilities.append(probability)

        }

        // ④ 完成した配列を返す
        return probabilities

    }

}
