import Foundation

struct ScaledScoreGradient {

    func calculate(

        scaledScoreGradient: Float,

        dimension: Int

    ) -> Float {

        let scale = sqrt(

            Float(dimension)

        )

        let scoreGradient =
            scaledScoreGradient / scale

        return scoreGradient
    }
}
