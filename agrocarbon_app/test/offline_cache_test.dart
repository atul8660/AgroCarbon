import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:agrocarbon_app/services/local_persistence_service.dart';
import 'package:agrocarbon_app/models/analysis_result.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  
  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  group('Offline Local Persistence Validations', () {
    test('Saving and restoring JSON correctly maps internal struct natively', () async {
      final mock = AnalysisResult(
        species: "Testing Species", 
        speciesConfidence: 0.9, health: "Healthy", healthConfidence: 0.9, 
        predictedBiomassKg: 100, carbonStoredKg: 50, co2EquivalentKg: 180, 
        estimatedCarbonCredit: 5, overallConfidence: 0.9
      );
      
      await LocalPersistenceService.savePrediction(mock);
      final restored = await LocalPersistenceService.restoreLastPrediction();
      
      expect(restored != null, true);
      expect(restored!.species, "Testing Species");
      expect(restored.carbonStoredKg, 50.0);
    });
  });
}
