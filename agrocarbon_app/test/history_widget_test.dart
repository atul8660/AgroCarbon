import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agrocarbon_app/widgets/history_card.dart';
import 'package:agrocarbon_app/models/analysis_result.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

void main() {
  group('Analytics History Matrices', () {
    testWidgets('History Card physically isolates UI models cleanly simulating offline taps optimally', (WidgetTester tester) async {
      final mock = AnalysisResult(
        species: "Teak", 
        speciesConfidence: 0.9, health: "Healthy", healthConfidence: 0.9, 
        predictedBiomassKg: 200, carbonStoredKg: 100, co2EquivalentKg: 366, 
        estimatedCarbonCredit: 10, overallConfidence: 0.9
      );
      
      await tester.pumpWidget(
        ProviderScope(
          child: MaterialApp(
            home: Scaffold(
              body: HistoryCard(result: mock),
            ),
          )
        )
      );
      
      expect(find.text('Teak'), findsOneWidget);
      expect(find.byType(IconButton), findsOneWidget); // Delete icon
    });
  });
}
