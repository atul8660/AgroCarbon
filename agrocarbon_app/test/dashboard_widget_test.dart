import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:agrocarbon_app/screens/main_navigation.dart';

void main() {
  testWidgets('MainNavigation renders BottomNavigationBar correctly', (WidgetTester tester) async {
    await tester.pumpWidget(
      const MaterialApp(
        home: MainNavigation(
          child: Scaffold(body: Text("Mock Child Shell")),
        ),
      )
    );
    expect(find.byType(NavigationBar), findsOneWidget);
    expect(find.text("Home"), findsOneWidget);
    expect(find.text("Profile"), findsOneWidget);
    expect(find.text("Settings"), findsOneWidget);
  });
}
