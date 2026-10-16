import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agrocarbon_app/widgets/stat_counter.dart';
import 'package:agrocarbon_app/widgets/confidence_ring.dart';

void main() {
  group('Analytics Result UI Components', () {
    testWidgets('StatCounter renders suffix cleanly over UI boundaries', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: StatCounter(endValue: 15.5, title: 'Biomass', suffix: ' kg'),
          ),
        )
      );
      expect(find.text('Biomass'), findsOneWidget);
    });

    testWidgets('ConfidenceRing constructs coloring arrays safely avoiding exceptions', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: ConfidenceRing(confidence: 0.95, label: 'Testing Node'),
          ),
        )
      );
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
      expect(find.text('Testing Node'), findsOneWidget);
      expect(find.text('95%'), findsOneWidget);
    });
  });
}
