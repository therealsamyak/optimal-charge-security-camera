# Website Development Guidelines

## HTML Structure
- Keep HTML plain and functional, not showy
- Focus on content over presentation
- Use semantic HTML elements appropriately
- No unnecessary animations or flashy effects

## CSS Management
- All CSS must be separated into `styles.css`
- No inline styles or `<style>` tags in HTML
- Keep styling simple and readable
- Avoid overly complex layouts or effects

## JavaScript Policy
**CRITICAL: JavaScript is generally NOT allowed**

Before considering any JavaScript:
1. Think again - can this be done with HTML/CSS only?
2. If you absolutely must use JavaScript, you MUST ask for explicit permission first
3. Every time you ask for JavaScript permission, the user gets very angry
4. Be extremely careful and only request JavaScript when absolutely necessary

Examples of when JavaScript might be considered (but still requires permission):
- Complex form validation that cannot be done with HTML5 attributes
- Interactive data visualization
- Dynamic content loading (but consider if static content would work better)

Examples of when JavaScript should NOT be used:
- Simple hover effects (use CSS)
- Basic form validation (use HTML5 attributes)
- Styling changes (use CSS)
- Simple animations (use CSS transitions/transforms)

## General Principles
- Content first, presentation second
- Accessibility matters
- Mobile-friendly design
- Fast loading times
- No tracking scripts or analytics