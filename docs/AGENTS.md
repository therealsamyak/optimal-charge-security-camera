# Website Development Guidelines

## Documentation Structure

- **Main page** (`index.html`): Project overview, goals, quick start guide
- **Documentation page** (`docs.html`): CLI reference and implementation details
- **Styling**: Custom CSS in `styles.css` for responsive layout
- **Navigation**: Top navigation bar + sidebar table of contents
- **JavaScript**: Minimal functionality in `script.js` for navigation

## HTML Structure

- Keep HTML plain and functional, not showy
- Focus on content over presentation
- Use semantic HTML elements appropriately
- No unnecessary animations or flashy effects
- Use custom CSS classes for layout and styling

## CSS Management

- All custom CSS must be in `styles.css`
- No inline styles or `<style>` tags in HTML
- Use custom CSS for responsive design and layout
- Keep styling simple and readable
- Avoid overly complex layouts or effects
- Use flexbox/grid for proper layout control

## JavaScript Policy

**JavaScript is allowed for basic navigation functionality**

Allowed JavaScript uses:

- Navigation highlighting and smooth scrolling
- Active link management
- Table of contents highlighting
- Basic user interaction improvements

JavaScript must be in `script.js` file only.

## File Organization

- `index.html`: Main landing page with project overview and quick start
- `docs.html`: Comprehensive documentation (CLI + Implementation)
- `styles.css`: All custom CSS styles for layout and design
- `script.js`: JavaScript for navigation functionality
- `CLI.md` and `IMPLEMENTATION.md`: Legacy markdown files (can be removed)
- All navigation links should use relative paths for local development

## Layout Structure

- **Top Navigation**: Fixed header with logo and page links
- **Sidebar**: Table of contents for current page only
- **Main Content**: Wide content area with proper width control
- **Responsive**: Mobile-friendly design with sidebar hidden on small screens

## Deployment Notes

- Website deployed at: https://therealsamyak.github.io/optimal-charge-security-camera/
- Internal navigation uses relative paths
- External links use full URLs
- All assets (CSS, JS) are self-contained

## General Principles

- Content first, presentation second
- Accessibility matters
- Mobile-friendly design
- Fast loading times
- No tracking scripts or analytics
- Clean, maintainable code structure
