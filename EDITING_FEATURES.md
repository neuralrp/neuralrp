# Character and World Info Editing Features

This document provides a comprehensive guide to the editing capabilities available in NeuralRP.

## Overview

NeuralRP now supports comprehensive editing of both character cards and world info entries, with AI-assisted editing capabilities that can analyze your current chat context or manual input to generate and update content.

## Character Card Editing

### Accessing Character Editing

1. Click the **Characters** button in the sidebar header
2. Click the **Edit** button (pencil icon) on any character card to enter edit mode
3. The character editing interface will appear with all fields available for editing

### Manual Editing

All character card fields can be edited manually:

- **Name**: The character's name
- **Label Description**: Personal notes about the character
- **Danbooru Tag**: Image generation tags
- **Description/Dialogue**: Character description and example dialogue
- **Reinforcement Prompt**: Context for the character's personality
- **Multi-Character Capsule**: Compact description used when multiple characters are active

### AI-Assisted Editing

#### Context Modes

Choose between two context modes for AI generation:

1. **Use Current Chat**: Uses the last 20 messages from your current conversation
2. **Manual Input**: Allows you to provide custom context text

#### Editable Fields with AI

Click any of these buttons to generate or update content:

- **Personality**: Generate personality traits and characteristics
- **Body**: Generate physical description and appearance details
- **Scenario**: Generate the current situation or setting
- **Genre**: Generate genre tags and themes
- **Tags**: Generate descriptive tags
- **First Message**: Generate an opening message for the character

#### Multi-Character Capsule

Two options for capsule generation:

1. **AI Generate**: Uses current chat context or manual input
2. **Auto Generate**: Automatically creates a capsule from existing character data

### Saving Characters

- Click **Save** to save all changes
- Click **Cancel** to discard changes and exit edit mode

## World Info Editing

### Accessing World Info Editing

1. Click the **World Info** button in the sidebar header
2. Expand a world info entry by clicking its name
3. Click the **Edit** button (pencil icon) on any entry to enter edit mode

### Manual Editing

Edit all world info entry fields:

- **Keys**: Comma-separated keywords for triggering this entry
- **Comment**: Optional descriptive comment
- **Content**: The main lore content
- **Canon Law**: Mark as immutable lore
- **Use Probability**: Enable probability-based triggering
- **Probability**: Set trigger probability (0-100)

### AI-Assisted Editing

#### Context Modes

Same options as character editing:

1. **Use Current Chat**: Uses last 50 messages for broader context
2. **Manual Input**: Custom context text

#### Editable Sections

Generate or update specific sections of world lore:

- **History**: Historical events and background
- **Locations**: Places, cities, and geographical features
- **Creatures**: Beings, monsters, and inhabitants
- **Factions**: Organizations, groups, and societies

### Managing World Info Entries

- **Add Entry**: Create new world info entries
- **Delete Entry**: Remove entries (with confirmation)
- **Toggle Canon Law**: Mark/unmark entries as immutable

## Advanced Features

### World Info Cache Management

Access cache controls in the Settings sidebar:

- **View Cache Stats**: See current cache usage and size
- **Clear Cache**: Remove all cached entries (may temporarily slow performance)
- **Configure Cache Size**: Set maximum number of cached entries (0 = unlimited)

### Performance Optimization

- **Smart Performance Mode**: Automatically queues LLM/SD operations
- **Adaptive Connection Monitoring**: Adjusts connection checks based on stability
- **Graceful Degradation**: Features disable automatically when services are unavailable

### Connection Management

Monitor and manage your AI service connections:

- **KoboldCpp**: For text generation (LLM)
- **Stable Diffusion**: For image generation
- **Real-time Status**: Live connection status and latency
- **Auto-reconnection**: Automatic retry with adaptive intervals

## Best Practices

### Character Editing

1. **Use Chat Context**: For most accurate generation, use current chat context
2. **Manual Input**: When you have specific details to include
3. **Iterative Editing**: Generate, review, and refine multiple times
4. **Multi-Character Capsules**: Keep them concise (2-3 sentences) for best performance

### World Info Editing

1. **Specific Sections**: Focus on one section at a time for better results
2. **Rich Context**: Use detailed chat history or manual input
3. **Canon Law**: Reserve for truly immutable lore elements
4. **Probability**: Use for optional or conditional lore elements

### Performance Tips

1. **Cache Management**: Regularly clear cache if you notice slowdowns
2. **Connection Monitoring**: Keep services online for full functionality
3. **Context Length**: Be mindful of context length for optimal performance

## Troubleshooting

### Common Issues

1. **AI Generation Fails**: Check service connections and try manual input
2. **Slow Performance**: Clear cache or reduce world info entries
3. **Connection Problems**: Use the test buttons to diagnose issues
4. **Missing Features**: Ensure services are connected (green status indicators)

### Error Messages

- **"Backend not running"**: Check that main.py is running
- **"Connection failed"**: Verify service URLs and network connectivity
- **"Invalid input"**: Check required fields and format
- **"Cache full"**: Increase cache size or clear existing entries

## Integration with Existing Features

### Chat Integration

- Editing uses current chat context for relevant generation
- Changes apply immediately to ongoing conversations
- World info updates affect future chat sessions

### Generation Features

- Character editing integrates with card generation
- World info editing works with world evolution
- Capsule generation uses edited character data

### Branch Management

- Edited content is preserved when creating branches
- Changes in one branch don't affect others
- World info is shared across all branches of a chat

## API Endpoints

For developers, the following endpoints are available:

### Character Editing
- `POST /api/characters/edit-field-ai`: Edit character fields with AI
- `POST /api/characters/edit-capsule-ai`: Edit multi-character capsules
- `POST /api/card-gen/generate-capsule`: Auto-generate capsules

### World Info Editing
- `POST /api/world-info/edit-entry`: Update world info entries
- `POST /api/world-info/add-entry`: Create new entries
- `POST /api/world-info/delete-entry`: Remove entries
- `POST /api/world-info/edit-entry-ai`: AI-assisted world info editing

### Cache Management
- `GET /api/world-info/cache/stats`: Get cache statistics
- `POST /api/world-info/cache/clear`: Clear cache
- `POST /api/world-info/cache/configure`: Configure cache settings

## Support

For additional help:
- Check the connection status indicators
- Review error messages in the browser console
- Ensure all required services are running
- Consult the main NeuralRP documentation for setup instructions