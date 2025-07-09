# GPU Kernel Examples Data

This directory contains XML data files for GPU kernel examples displayed in the GUI application.

## Structure

- `examples/` - Contains XML files for each kernel example
- `examples/examples_list.xml` - Master list of all available examples

## XML Format

Each example XML file follows this structure:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<example>
    <name>Example Name</name>
    <category>Basic|Advanced</category>
    <sourceFile>relative/path/to/source.cu</sourceFile>
    <description>
        <title>Example Title</title>
        <analogy>Optional analogy section (rich text)</analogy>
        <overview>Overview text</overview>
        <features>
            <feature>Feature description</feature>
            <!-- more features -->
        </features>
        <concepts>
            <concept>
                <title>Concept Name</title>
                <description>Concept description</description>
            </concept>
            <!-- more concepts -->
        </concepts>
        <applications>
            <application>
                <title>Application Name</title>
                <description>Application description</description>
            </application>
            <!-- more applications -->
        </applications>
        <optimizations>
            <optimization>Optimization description</optimization>
            <!-- more optimizations -->
        </optimizations>
        <patterns>
            <pattern>Pattern description</pattern>
            <!-- more patterns -->
        </patterns>
        <algorithms>
            <algorithm>Algorithm description</algorithm>
            <!-- more algorithms -->
        </algorithms>
        <performance>
            <consideration>Performance consideration</consideration>
            <!-- more considerations -->
        </performance>
        <importance>
            <why>Why this matters explanation</why>
            <performance>Performance impact explanation</performance>
        </importance>
    </description>
</example>
```

## Benefits of XML-Based System

1. **Maintainability**: Easy to edit descriptions without recompiling
2. **Internationalization**: Can add multiple language files
3. **Consistency**: Structured format ensures uniform presentation
4. **Extensibility**: Easy to add new fields or sections
5. **Validation**: XML can be validated against schemas
6. **Version Control**: Text files work well with git

## Adding New Examples

1. Create a new XML file in the `examples/` directory
2. Follow the naming convention: `example_name.xml`
3. Add the file reference to `examples_list.xml`
4. The GUI will automatically load the new example

## Editing Descriptions

Simply edit the XML files and restart the GUI. No recompilation needed!

## Fallback Behavior

If XML files cannot be loaded, the application falls back to hardcoded descriptions to ensure functionality.
