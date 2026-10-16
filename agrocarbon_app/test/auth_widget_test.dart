import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agrocarbon_app/widgets/custom_text_field.dart';
import 'package:agrocarbon_app/widgets/primary_button.dart';

void main() {
  testWidgets('PrimaryButton renders Loading Spinner successfully', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: PrimaryButton(text: 'Login', onPressed: _dummy, isLoading: true),
        )
      )
    );
    expect(find.byType(CircularProgressIndicator), findsOneWidget);
    expect(find.text('Login'), findsNothing);
  });

  testWidgets('CustomTextField renders input bounds securely', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: CustomTextField(label: 'Secure Password', isPassword: true),
        )
      )
    );
    expect(find.text('Secure Password'), findsOneWidget);
  });
}

void _dummy() {}
