import 'package:flutter_test/flutter_test.dart';
import 'package:agrocarbon_app/models/analysis_result.dart';

void main() {
  group('AI Upload Analytics JSON Mappings', () {
    test('AnalysisResult accurately un-marshals AI output matrices securely', () {
      final payload = {
        'species': 'Teak',
        'species_confidence': 0.99,
        'health': 'Healthy',
        'health_confidence': 0.95,
        'predicted_biomass_kg': 150.0,
        'carbon_stored_kg': 75.0,
        'co2_equivalent_kg': 275.0,
        'estimated_carbon_credit': 0.25,
        'overall_confidence': 0.96,
      };

      final result = AnalysisResult.fromJson(payload);

      expect(result.species, 'Teak');
      expect(result.carbonStoredKg, 75.0);
      expect(result.overallConfidence, 0.96);
    });
  });
}
